import numpy as np
from functools import reduce
from math import gcd, gamma
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import scipy.ndimage as ndi
import heapq


def metric_axes(aspect, ndim):
    """Determinant-normalised half-axes, so vol{d <= w} is the same at any aspect."""
    n = float(ndim)
    ra = float(aspect) ** ((n - 1.0) / n)
    return ra, float(aspect) ** (-1.0 / n)


def distance_uniform(shape, seeds, axis, aspect):
    """Closed form for a constant director. No graph, no sweeps, no error.

        d(x) = sqrt( (e.u / r_a)^2 + |e - (e.u)u|^2 / r_b^2 ),   e = x - seed

    `seeds` are index tuples in array order; `axis` is in the same order.
    """
    ndim = len(shape)
    ra, rb = metric_axes(aspect, ndim)
    u = np.asarray(axis, np.float32)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    grid = np.stack(np.mgrid[tuple(slice(0, s) for s in shape)]).astype(np.float32)
    d = np.full(shape, np.inf, np.float32)
    for s in seeds:
        e = grid - np.asarray(s, np.float32).reshape((ndim,) + (1,) * ndim)
        par = np.einsum("i,i...->...", u, e)
        per2 = np.maximum((e * e).sum(0) - par * par, 0.0)
        np.minimum(d, np.sqrt((par / ra) ** 2 + per2 / (rb * rb)), out=d)
    return d

def stencil(ndim, aspect):
    """Neighbour offsets for the geodesic graph: every PRIMITIVE integer offset within a
    radius. """
    r = 3 if (ndim == 2 and aspect >= 1.6) else 2
    out = []
    for d in np.ndindex(*(2 * r + 1,) * ndim):
        v = tuple(int(x) - r for x in d)
        if all(x == 0 for x in v):
            continue
        if reduce(gcd, [abs(x) for x in v]) == 1:      # primitive only
            out.append(v)
    return out

def anisotropic_graph(director, aspect, st=None):
    """Sparse neighbour graph whose edge weights follow the local director.

    Edge p->q is priced as the MEAN of the step cost at p and at q. That makes the graph
    symmetric, so one undirected Dijkstra gives an exact lattice geodesic, and multi-source
    is free (`min_only=True` returns the minimum over all seeds in a single pass).
    """
    ndim = director.shape[0]
    shape = director.shape[1:]
    st = stencil(ndim, aspect) if st is None else st
    ra, rb = metric_axes(aspect, ndim)
    n = int(np.prod(shape))
    idx = np.arange(n).reshape(shape)
    rows, cols, data = [], [], []
    for off in st:
        e = np.asarray(off, np.float32)
        par = np.einsum("i,i...->...", e, director)
        per2 = np.maximum(float(e @ e) - par * par, 0.0)
        cost = np.sqrt((par / ra) ** 2 + per2 / (rb * rb)).astype(np.float32)
        src = tuple(slice(max(0, -o), s - max(0, o)) for o, s in zip(off, shape))
        dst = tuple(slice(max(0, -o) + o, s - max(0, o) + o) for o, s in zip(off, shape))
        if any(sl.stop <= sl.start for sl in src):
            continue
        rows.append(idx[src].ravel())
        cols.append(idx[dst].ravel())
        data.append((0.5 * (cost[src] + cost[dst])).ravel())
    g = coo_matrix((np.concatenate(data).astype(np.float64),
                    (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return g.tocsr()

def distance_geodesic(graph, seeds, shape):
    """Exact multi-source lattice geodesic; d = min over this structure's own seeds."""
    flat = [int(np.ravel_multi_index(s, shape)) for s in seeds]
    d = dijkstra(graph, directed=False, indices=flat, min_only=True)
    return d.reshape(shape).astype(np.float32)

def distance_field(shape, seeds, aspect, director=None, axis=None, downsample=2,
                   graph=None):
    """The distance field for one structure, by whichever route is right."""
    if director is None:
        return distance_uniform(shape, seeds, axis, aspect)
    f = max(int(downsample), 1)
    if f == 1:
        g = anisotropic_graph(director, aspect) if graph is None else graph
        return distance_geodesic(g, seeds, shape)
    small = tuple(max(s // f, 2) for s in shape)
    u = np.stack([ndi.zoom(director[a], [t / s for t, s in zip(small, shape)], order=1,
                           mode="nearest") for a in range(director.shape[0])])
    u = u / np.maximum(np.linalg.norm(u, axis=0, keepdims=True), 1e-12)
    sd = [tuple(int(np.clip(c * t // s, 0, t - 1)) for c, t, s in zip(p, small, shape))
          for p in seeds]
    g = anisotropic_graph(u.astype(np.float32), aspect) if graph is None else graph
    d = distance_geodesic(g, sd, small)
    up = ndi.zoom(d, [s / t for s, t in zip(shape, small)], order=1, mode="nearest",
                  grid_mode=True)
    return (up * float(f)).astype(np.float32)


def place_seeds(shape, counts, draws, spread=1.0):
    """Best-candidate ("Mitchell") sampling: each new seed is the FARTHEST of a batch of
    frozen candidates from every seed already placed. 
    This is a selction from frozen draws"""
    shape = np.asarray(shape, float)
    ndim = len(shape)
    cand = (0.5 + (np.asarray(draws, float)[:, :ndim] - 0.5) * float(spread)) * shape
    cand = np.clip(np.rint(cand), 0, shape - 1).astype(int)
    out, placed, cur = [], [], 0
    for k in counts:
        per = []
        for _ in range(int(k)):
            batch = cand[cur:cur + 16]
            cur += 16
            if batch.size == 0:
                break
            if placed:
                d2 = ((batch[:, None, :] - np.asarray(placed)[None]) ** 2).sum(-1).min(1)
                p = batch[int(np.argmax(d2))]
            else:
                p = batch[0]
            placed.append(tuple(p))
            per.append(tuple(int(v) for v in p))
        out.append(per)
    return out

def _prune_to_seeds(labels, seeds, conn):
    """Release any claimed voxel not face-connected to one of its OWN seeds.
    """
    vols = np.zeros(len(seeds))
    for i, sd in enumerate(seeds):
        m = labels == i
        if not m.any():
            continue
        cc, _ = ndi.label(m, structure=conn)
        keep = {int(cc[s]) for s in sd if cc[s] > 0}
        good = np.isin(cc, list(keep)) if keep else np.zeros_like(m)
        labels[m & ~good] = -1
        vols[i] = float(good.sum())
    return vols

def _shell_volumes(labels, stack, w, holes, k):
    """Volume that COUNTS toward the target.
    For a solid structure that is its whole territory. For a hollow one like a vessel the
    lumen is inside the territory and is claimed, so nothing leaks into it, but it is NOT
    tissue and must count zero."""
    m = labels == k
    if holes is None or holes[k] <= 0.0:
        return float(m.sum())
    return float((m & (stack[k] > holes[k] * w[k])).sum())

def _assign(stack, w, seeds, conn, mask, holes=None):
    val = stack - w.reshape((-1,) + (1,) * (stack.ndim - 1)).astype(np.float32)
    arg = np.argmin(val, axis=0).astype(np.int16)
    best = np.take_along_axis(val, arg[None], axis=0)[0]
    labels = np.where(best < 0, arg, np.int16(-1)).astype(np.int16)
    if mask is not None:
        labels[~mask] = -1
    _prune_to_seeds(labels, seeds, conn)          # the LUMEN stays part of the body
    vols = np.array([_shell_volumes(labels, stack, w, holes, k) for k in range(len(seeds))])
    return labels, vols


def _grow_to_targets(labels, vols, stack, w, targets, conn, mask):
    """Greedy priority growth until every structure hits its target.
    Prune Voxels that got separated from their seed structure"""
    need = {i: int(round(targets[i] - vols[i])) for i in range(len(targets))
            if targets[i] - vols[i] >= 1}
    if not need:
        return labels, vols
    shape = labels.shape
    flat = labels.reshape(-1)
    heap = []
    for i in need:
        m = labels == i
        fr = ndi.binary_dilation(m, structure=conn) & ~m & (labels == -1)
        if mask is not None:
            fr &= mask
        p = np.flatnonzero(fr.ravel())
        key = (stack[i].ravel()[p] - w[i]).astype(float)
        heap.extend(zip(key.tolist(), [i] * p.size, p.tolist()))
    heapq.heapify(heap)
    steps = [int(np.prod(shape[a + 1:])) for a in range(len(shape))]
    while heap and need:
        _, i, p = heapq.heappop(heap)
        if i not in need or flat[p] != -1:
            continue
        flat[p] = i
        vols[i] += 1
        need[i] -= 1
        if need[i] <= 0:
            del need[i]
            continue
        coord = np.unravel_index(p, shape)
        for a, s in enumerate(steps):
            for d in (-1, 1):
                c = coord[a] + d
                if not (0 <= c < shape[a]):
                    continue
                q = p + d * s
                if flat[q] == -1 and (mask is None or mask.reshape(-1)[q]):
                    heapq.heappush(heap, (float(stack[i].reshape(-1)[q] - w[i]), i, int(q)))
    return labels, vols

def lic_smooth(field, director, half_len, step=1.0):
    """Average `field` along the streamlines of `director`.
    """
    shape = field.shape
    ndim = len(shape)
    grid = np.stack(np.mgrid[tuple(slice(0, s) for s in shape)]).astype(np.float64)
    acc = field.astype(np.float64).copy()
    cnt = np.ones_like(acc)
    n_steps = max(1, int(round(half_len / step)))
    hi = np.asarray(shape, float).reshape((ndim,) + (1,) * ndim) - 1.0
    for sgn in (1.0, -1.0):
        p = grid.copy()
        d = sgn * director.astype(np.float64)
        alive = np.ones(shape, bool)
        for _ in range(n_steps):
            p = p + d * step
            alive &= np.all((p >= 0) & (p <= hi), axis=0)
            if not alive.any():
                break
            ix = tuple(np.clip(p[a], 0, shape[a] - 1).astype(np.intp) for a in range(ndim))
            acc += np.where(alive, field[ix], 0.0)
            cnt += alive
            nxt = director[(slice(None),) + ix].astype(np.float64)
            d = np.where((nxt * d).sum(0) < 0, -nxt, nxt)
    out = acc / cnt
    return ((out - out.mean()) / (out.std() + 1e-12)).astype(np.float32)

def carve_to_coverage(labels, coverage, score_extra=None, margin=2.0, hole=None):
    """Remove random nematic field until tissue occupies exactly round(coverage*N) voxels.

    Voxels are ranked by distance from the nearest structure (plus whatever `score_extra`
    adds), and the worst-ranked are dropped.
    The margin guarantees any voxel within `margin` of
    a structure is unconditionally kept, so every structure keeps a stromal rim.

    Returns (tissue, dist, score). Raises if the structures plus their collars already
    exceed the coverage budget, rather than silently eating the collar.
    """
    shape = labels.shape
    n_tot = int(np.prod(shape))
    n_keep = int(round(float(coverage) * n_tot))
    struct = labels >= 0
    hole = np.zeros(shape, bool) if hole is None else np.asarray(hole, bool)
    solid = struct & ~hole
    dist = ndi.distance_transform_edt(~struct)
    score = dist.astype(np.float64)
    if score_extra is not None:
        score = score + np.asarray(score_extra, np.float64)
    protected = solid | ((dist <= float(margin)) & ~hole)
    n_prot = int(protected.sum())
    if n_prot > n_keep:
        raise ValueError(
            f"infeasible: the structures plus a {margin:g}-voxel collar need {n_prot} "
            f"voxels ({n_prot / n_tot:.1%} of the frame) but COVER allows {n_keep} "
            f"({coverage:.1%}). Lower a FRAC, lower MARGIN_UM, or raise COVER.")
    if n_keep > n_tot - int(hole.sum()):
        raise ValueError(
            f"infeasible: COVER asks for {n_keep} tissue voxels but {int(hole.sum())} are "
            f"lumen, leaving only {n_tot - int(hole.sum())}. Lower COVER, lower a vessel's "
            f"FRAC, or raise WALL_IN so its lumen is smaller.")
    score[protected] = -np.inf
    score[hole] = np.inf                       # a lumen is never tissue
    keep = np.argpartition(score.ravel(), n_keep - 1)[:n_keep]
    tissue = np.zeros(n_tot, bool)
    tissue[keep] = True
    return tissue.reshape(shape), dist, score


def pressure_pack(shape, seeds, targets, distances, mask=None, holes=None, damping=0.9,
                  tol=2e-3, max_iter=250, stall_patience=30):
    """Inflate each structure until it occupies exactly `targets[i]` voxels.

    Returns (labels, weights, volumes, iterations, converged)."""
    ndim = len(shape)
    conn = ndi.generate_binary_structure(ndim, 1)
    targets = np.asarray(targets, float)
    stack = np.stack([np.asarray(d, np.float32) for d in distances])
    kk = np.array([max(len(s), 1) for s in seeds], float)

    # start at the free-space radius: the answer if the structure had no neighbours
    ball = np.pi ** (ndim / 2.0) / gamma(ndim / 2.0 + 1.0)
    w = (np.maximum(targets, 1.0) / ball) ** (1.0 / ndim)
    cap = 0.25 * w + 1.0
    n_tot = float(np.prod(shape))
    labels = np.full(shape, -1, np.int16)
    vols = np.zeros(len(targets))
    best, last, it, converged = np.inf, 0, 0, False

    for it in range(1, max_iter + 1):
        labels, vols = _assign(stack, w, seeds, conn, mask, holes)
        err = float(np.abs(targets - vols).max())
        if err / n_tot < tol:
            converged = True
            break
        # dV/dw is the interface AREA, so dividing the volume error by it turns the error
        # into a step in RADIUS units
        surf = np.maximum(ndim * ball ** (1.0 / ndim)
                          * np.maximum(vols, 1.0) ** ((ndim - 1.0) / ndim)
                          * kk ** (1.0 / ndim), 6.0)
        w += np.clip(damping * (targets - vols) / surf, -cap, cap)
        if err < best - 5e-4 * n_tot:
            best, last = err, it
        elif it - last > stall_patience:
            break
    for _ in range(40):
        over = vols > targets
        if not over.any():
            break
        w[over] -= np.maximum(0.35, 0.015 * w[over])
        labels, vols = _assign(stack, w, seeds, conn, mask, holes)

    labels, vols = _grow_to_targets(labels, vols, stack, w, targets, conn, mask)
    return labels, w, vols, it, converged


def enforce_cohesion(tissue, score, min_island, protected):
    """Dissolve tissue islands below `min_island` voxels and regrow the SAME number onto
    the surviving tissue by priority flood, so coverage stays exact to the voxel.
    """
    conn = ndi.generate_binary_structure(tissue.ndim, 1)
    cc, n = ndi.label(tissue, structure=conn)
    if n <= 1:
        return tissue
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    keep = set(np.flatnonzero(sizes >= min_island).tolist())
    keep.add(int(np.argmax(sizes)))
    keep |= {int(v) for v in np.unique(cc[protected & (cc > 0)])}
    doomed = tissue & ~np.isin(cc, list(keep))
    budget = int(doomed.sum())
    if budget == 0:
        return tissue
    tissue = tissue & ~doomed
    shape = tissue.shape
    flat_t = tissue.reshape(-1)
    flat_s = np.asarray(score, float).reshape(-1)
    queued = np.zeros(tissue.size, bool)
    fr = ndi.binary_dilation(tissue, structure=conn) & ~tissue
    p = np.flatnonzero(fr.ravel())
    queued[p] = True
    heap = list(zip(flat_s[p].tolist(), p.tolist()))
    heapq.heapify(heap)
    steps = [int(np.prod(shape[a + 1:])) for a in range(len(shape))]
    while budget > 0 and heap:
        _, p = heapq.heappop(heap)
        if flat_t[p]:
            continue
        flat_t[p] = True
        budget -= 1
        coord = np.unravel_index(p, shape)
        for a, s in enumerate(steps):
            for d in (-1, 1):
                c = coord[a] + d
                if 0 <= c < shape[a]:
                    q = p + d * s
                    if not flat_t[q] and not queued[q]:
                        queued[q] = True
                        heapq.heappush(heap, (float(flat_s[q]), int(q)))
    return tissue
