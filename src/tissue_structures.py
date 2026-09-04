
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi

import tissue_direction as td
import pressure_pack as pp
from scene import _smooth_z


@dataclass
class StructureCtx:
    """Everything one tissue structure's hooks may look at, bundled so each hook takes a single
    argument -- the NoiseCtx pattern, one layer up."""

    shape: tuple
    spacing: tuple
    um_per_vox: float
    # prior director, array order (3, z, y, x)
    u: np.ndarray
    # this structure's own seed index-tuples
    seeds: list

    # the (blended) director distance() grew under
    grow_dir: np.ndarray | None = None
    # this structure's own distance field, un-shifted
    dist_raw: np.ndarray | None = None
    # its pressure weight w_i (the radius it settled at)
    level: float | None = None
    # labels == i
    territory: np.ndarray | None = None
    # its soft membership (a partition-of-unity slice)
    m_i: np.ndarray | None = None

    def px(self, p):
        """A `P` holding microns -> lateral voxels, matching NoiseCtx.px."""
        return max(float(p.v) / float(self.um_per_vox), 1e-6)

    def require(self, *names):
        """Guard a FIELD-pass hook against being run before the pressure solve."""
        miss = [k for k in names if getattr(self, k) is None]
        if miss:
            raise RuntimeError(
                f"StructureCtx.{miss[0]} is None -- this hook runs in the FIELD pass but was "
                f"called before the pressure solve. build_architecture fills "
                f"dist_raw / level / territory / m_i only after pressure_pack().")


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)

def _axis_of(st):
    """A structure's declared direction as a unit vector in ARRAY order (z, y, x).
    """
    t, d = np.deg2rad(float(st.TILT_DEG.v)), np.deg2rad(float(st.DIR_DEG.v))
    return _unit([np.cos(t), np.sin(t) * np.sin(d), np.sin(t) * np.cos(d)])

def _structure_distance(st, ctx):
    """The default growth metric behind BaseStructure.distance: the field whose level sets are
    the shapes this structure can settle into. Either a uniform anisotropic metric along the
    declared axis (FLOW='axis', closed form, no solve) or the geodesic under the prior grain
    slerped toward that axis (FLOW='field'). It records the director it grew under on
    ctx.grow_dir so director() reads back the same field.
    """
    asp = float(st.ASPECT.v)
    if getattr(st, "FLOW", "field") == "axis":
        ctx.grow_dir = None
        return pp.distance_uniform(ctx.shape, ctx.seeds, _axis_of(st), asp)
    # blend the prior flow toward this structure's declared direction first, so the metric it
    # grows under and the director it reports are one field
    ctx.grow_dir = blend_axis(ctx.u, _axis_of(st), float(st.SIMILARITY.v))
    return pp.distance_field(ctx.shape, ctx.seeds, asp, director=ctx.grow_dir, downsample=2)


def _roughen(d, st, noise_i, k_i, cover, n_tot, um_per_vox):
    """Size-invariant boundary wobble: perturb the distance field rather than the pressure, so
    rough scales it smoothly and the structure's volume still comes out exact. Generic, so a
    structure that overrides distance() still gets it.
    """
    r = float(st.ROUGH.v)
    if r <= 0.0:
        return d.astype(np.float32)
    rad = (float(st.FRAC.v) * cover * n_tot / max(k_i, 1) * 3.0 / (4.0 * np.pi)) ** (1 / 3)
    sg = max(float(st.BUMP_SCALE_UM.v) / float(um_per_vox), 0.5)
    xi = np.clip(_smooth_z(np.asarray(noise_i, np.float32), sg, sg), -2.5, 2.5)
    return (d + np.float32(r * 0.55 * rad) * xi).astype(np.float32)


def _director_ordered(st, ctx):
    """An ordered bundle points along the metric it actually grew under: the declared axis if it
    grew uniformly, the blended grain if it grew under the field.
    """
    if getattr(st, "FLOW", "field") == "axis":
        # (z, y, x)
        a = _axis_of(st)
        # -> (x, y, z)
        v = np.array([a[2], a[1], a[0]], np.float64)
        return np.broadcast_to(v[:, None, None, None], (3,) + tuple(ctx.shape))
    u = ctx.grow_dir if ctx.grow_dir is not None else ctx.u
    return u[[2, 1, 0]].astype(np.float64)


def _director_vessel(st, ctx):
    """A duct wall frame, straight from the structure's own distance field.
    """
    ctx.require("dist_raw")
    gz, gy, gx = np.gradient(ctx.dist_raw, *ctx.spacing)
    er = td.unit(np.stack([gx, gy, gz]))
    a = _axis_of(st)                                  # (z, y, x)
    ea = np.asarray([a[2], a[1], a[0]], np.float64)[:, None, None, None]
    ea = td.unit(ea - (ea * er).sum(0, keepdims=True) * er)
    ec = np.stack([ea[1] * er[2] - ea[2] * er[1],
                   ea[2] * er[0] - ea[0] * er[2],
                   ea[0] * er[1] - ea[1] * er[0]])
    ph, pt = np.deg2rad(float(st.PHASE_DEG.v)), np.deg2rad(float(st.PHASE_TILT_DEG.v))
    return np.cos(pt) * (np.cos(ph) * er + np.sin(ph) * ec) + np.sin(pt) * ea


def _lumen_ring(st, ctx):
    """A soft Gaussian band riding the lumen wall -- the surface an epithelium lines. Built from
    the structure's own distance field, so it hugs the true ragged boundary rather than a circle
    drawn around the seed. Width is st._RING_W as a fraction of the wall's own thickness.
    """
    ctx.require("dist_raw", "level")
    lum = float(st.WALL_IN.v) * ctx.level
    wall = max(st._RING_W * (ctx.level - lum), 1e-6)
    return np.exp(-(((ctx.dist_raw - lum) / wall) ** 2)).astype(np.float32)

def prior_noise(tape, ARCH, shape, um_per_vox):
    """The prior director as a Q-TENSOR, and its own weight map.
    """
    sxy = max(float(ARCH.SCALE_UM.v) / float(um_per_vox), 1e-6)
    sz_ = max(float(ARCH.SCALE_Z_UM.v) / float(um_per_vox), 1e-6)
    c = np.stack([_smooth_z(np.asarray(tape[k], np.float32), sxy, sz_)
                  for k in ("dir_a", "dir_b", "dir_c", "dir_d", "dir_e")])
    return td.noise6(c, max(float(ARCH.NOISE_W.v), 1e-12))


def prior_field(tape, ARCH, shape, um_per_vox):
    """The director that exists before any structure: the isotropic random flow, bent.
    """
    sxy = max(float(ARCH.SCALE_UM.v) / float(um_per_vox), 1e-6)
    sz_ = max(float(ARCH.SCALE_Z_UM.v) / float(um_per_vox), 1e-6)
    c = np.stack([_smooth_z(np.asarray(tape[k], np.float32), sxy, sz_)
                  for k in ("dir_a", "dir_b", "dir_c", "dir_d", "dir_e")])
    q, _ = td.noise6(c, max(float(ARCH.NOISE_W.v), 1e-6))
    n, _ = td.decompose(q)                      # (3, ...) as (x, y, z)
    if float(ARCH.CURV.v) != 0.0:
        # a global bend of the grain, deg/um, about the section normal
        nz, ny, nx = shape
        Y, X = np.mgrid[0:ny, 0:nx].astype(np.float64)
        a = np.deg2rad(float(ARCH.CURV.v)) * float(um_per_vox) * (X - 0.5 * (nx - 1))
        ca, sa = np.cos(a)[None], np.sin(a)[None]
        nx_, ny_ = n[0] * ca - n[1] * sa, n[0] * sa + n[1] * ca
        n = np.stack([nx_, ny_, n[2]])
    return np.stack([n[2], n[1], n[0]]).astype(np.float32)      # -> array order (z, y, x)

def seed_counts(structs, cover, n_tot, um_per_vox):
    """Decide how many separate structures (seeds) each structure is foing to spawn
    """
    out = []
    for st in structs:
        a = float(st.FRAC.v) * cover * n_tot
        v = 4.0 / 3.0 * np.pi * (float(st.SIZE_UM.v) / float(um_per_vox)) ** 3
        out.append(int(np.clip(round(a / max(v, 1.0)), 1, int(round(float(st.N_MAX.v))))))
    return out


def blend_axis(u, axis, s):
    """Slerp a whole director FIELD toward a single axis, nematically."""
    s = float(np.clip(s, 0.0, 1.0))
    v = _unit(axis).astype(np.float64)
    if s <= 0.0:
        return u
    if s >= 1.0:
        return np.broadcast_to(v[:, None, None, None], u.shape).copy()
    d = np.einsum("i,i...->...", v, u)
    # the near end of the headless axis
    un = np.where(d < 0.0, -u, u)                       
    g = np.arccos(np.clip(np.abs(d), -1.0, 1.0))
    k = np.stack([un[1] * v[2] - un[2] * v[1],
                  un[2] * v[0] - un[0] * v[2],
                  un[0] * v[1] - un[1] * v[0]])
    kn = np.linalg.norm(k, axis=0)
    k = np.where(kn > 1e-9, k / np.maximum(kn, 1e-12), 0.0)
    t = s * g
    ct, st_ = np.cos(t), np.sin(t)
    kxu = np.stack([k[1] * un[2] - k[2] * un[1],
                    k[2] * un[0] - k[0] * un[2],
                    k[0] * un[1] - k[1] * un[0]])
    kdu = np.einsum("i...,i...->...", k, un)
    out = un * ct + kxu * st_ + k * kdu * (1.0 - ct)
    return (out / np.maximum(np.linalg.norm(out, axis=0, keepdims=True), 1e-12)).astype(np.float32)


def build_architecture(tape, ARCH, TG, shape, spacing, um_per_vox, L=4, l_min=2):
    """Field -> seeds -> pressurised expansion -> carve. Returns (support, field, report)."""
    nz, ny, nx = shape
    n_tot = int(nz) * int(ny) * int(nx)
    cover = float(np.clip(TG.COVER.v, 0.0, 1.0))
    structs = list(getattr(ARCH, "Structures", []) or [])
    frac_sum = sum(float(s.FRAC.v) for s in structs)
    if frac_sum > 1.0 + 1e-9:
        raise ValueError(
            f"structure FRACs sum to {frac_sum:.3f}; they are shares of the TISSUE, so they "
            f"must leave room for the interstitium. Lower some, or raise TG.COVER.")

    report = {"targets": {}, "realised": {}, "levels": {}, "territory": {},
              "count": {}, "size_um": {}, "body_kept": {}}
    names = [getattr(s, "name", None) or f"structure{i}" for i, s in enumerate(structs)]

    # 1) the prior director, over the whole volume
    # `u` is the unit director structures grow along.
    q_prior, w_prior = prior_noise(tape, ARCH, shape, um_per_vox)
    u = prior_field(tape, ARCH, shape, um_per_vox)

    # 2) seeds: best-candidate selection over the frozen pool
    ks = seed_counts(structs, cover, n_tot, um_per_vox)
    seeds = []
    for i, st in enumerate(structs):
        seeds += pp.place_seeds(shape, [ks[i]], np.asarray(tape["struct_seed"][i], float),
                                 spread=float(np.clip(st.SPREAD.v, 0.05, 1.0)))

    ctxs = [StructureCtx(shape=shape, spacing=spacing, um_per_vox=um_per_vox,
                         u=u, seeds=seeds[i]) for i in range(len(structs))]

    # 3) distance fields: each structure names its own metric, then a generic size-invariant
    # roughening that keeps the volume exact
    dists = []
    for i, st in enumerate(structs):
        d = st.distance(ctxs[i])
        d = _roughen(d, st, tape["struct_noise"][i], ks[i], cover, n_tot, um_per_vox)
        dists.append(d.astype(np.float32))

    # 4) the pressure solve: exact volumes.
    walls = np.array([float(st.wall_fraction()) for st in structs])
    targets = np.array([float(st.FRAC.v) * cover * n_tot for st in structs])
    labels, w_i, vols, iters, converged = pp.pressure_pack(shape, seeds, targets, dists,
                                                            holes=walls)

    for i, c in enumerate(ctxs):
        c.dist_raw, c.level, c.territory = dists[i], float(w_i[i]), (labels == i)

    # 5) Random tissue wraps the structures
    extra = None
    if float(ARCH.EDGE_NOISE.v) > 0.0:
        sg = max(float(ARCH.EDGE_SCALE_UM.v) / float(um_per_vox), 0.5)
        xi = _smooth_z(np.asarray(tape["carve_noise"], np.float32), sg, sg)
        if float(ARCH.LIC_STRETCH.v) > 0.0:
            xi = pp.lic_smooth(xi, u, float(ARCH.LIC_STRETCH.v) * sg)
        extra = float(ARCH.EDGE_NOISE.v) * xi
    if float(ARCH.COMPACT.v) != 0.0:
        Y, X = np.mgrid[0:ny, 0:nx].astype(np.float32)
        rr = np.hypot((Y - 0.5 * (ny - 1)) / max(ny / 2.0, 1.0),
                      (X - 0.5 * (nx - 1)) / max(nx / 2.0, 1.0))
        bias = float(ARCH.COMPACT.v) * np.clip(rr, 0.0, 1.0)[None, :, :]
        extra = bias if extra is None else extra + bias
    margin = float(ARCH.MARGIN_UM.v) / float(um_per_vox)

    # 6) each structure's claimed-but-not-tissue interior (a vessel lumen): claimed so nothing
    # leaks in, but NOT tissue -- it does not spend the coverage budget and never gets a cell.
    # Computed once here, reused by the carve and the ledger.
    interior_i = []
    for i, st in enumerate(structs):
        mi = st.interior(ctxs[i])
        interior_i.append(np.zeros(shape, bool) if mi is None else np.asarray(mi, bool))
    interior = (np.logical_or.reduce(interior_i) if interior_i
                else np.zeros(shape, bool))

    support, dist_bg, score = pp.carve_to_coverage(labels, cover, score_extra=extra,
                                                    margin=margin, hole=interior)
    if float(ARCH.MIN_ISLAND.v) > 0.0:
        support = pp.enforce_cohesion(support, score,
                                       float(ARCH.MIN_ISLAND.v) * n_tot,
                                       (labels >= 0) | (dist_bg <= margin))

    # 7) membership: a softmin, a partition of unity by construction
    # No sequential discount. That existed only to repair the declaration-order overlap of
    # the old claim rule, and the pressure balance does not create any.
    keys = np.stack([(dists[i] - w_i[i])
                     / max(float(structs[i].EDGE_UM.v) / float(um_per_vox), 1e-3)
                     for i in range(len(structs))])
    ex = np.exp(-np.clip(keys, -60.0, 60.0))
    m = (ex / (1.0 + ex.sum(0, keepdims=True))).astype(np.float32)

    for i, c in enumerate(ctxs):
        c.m_i = m[i]

    # 8) the director: prior flow, the margin, then each structure's prescription
    Q6 = q_prior.astype(np.float32)
    resp = [w_prior.astype(np.float32)]
    w_marg = np.zeros(shape, np.float32)
    if float(ARCH.TANGENCY.v) > 0.0:
        sig = float(np.std(score[np.isfinite(score)])) + 1e-12
        gz, gy, gx = np.gradient(dist_bg, *spacing)
        a3 = np.arctan2(gy, gx) + 0.5 * np.pi
        tz = np.deg2rad(float(ARCH.TANGENCY_Z_DEG.v))
        n3 = np.stack([np.cos(tz) * np.cos(a3), np.cos(tz) * np.sin(a3),
                       np.sin(tz) + np.zeros_like(a3)])
        band = float(ARCH.BAND.v) * sig
        w_marg = (float(ARCH.TANGENCY.v) * np.exp(-((dist_bg / max(band, 1e-6)) ** 2))
                  * (1.0 - m.sum(0))).astype(np.float32)
        Q6 += td.uni6(n3, w_marg).astype(np.float32)
    resp.append(w_marg)

    recs = []
    for i, st in enumerate(structs):
        ws = float(st.W.v)
        if ws > 0.0:
            Q6 += td.uni6(st.director(ctxs[i]), ws * m[i]).astype(np.float32)
        resp.append((ws * m[i]).astype(np.float32))
        recs.append({
            "name": names[i],
            "kind": getattr(st, "kind", "?"),
            "level": float(w_i[i]), "wall": float(walls[i]),
            "seeds": seeds[i],
            "k": ks[i],
        })

    # 9) scalar features for cell typing (via CellType.Affinity) and for plotting. "S" is the
    # built-in coherence feature every architecture has; each structure adds its own. Same-name
    # fields from different structures combine by elementwise max.
    _, gap = td.decompose(Q6)
    features = {"S": td.coherence(gap).astype(np.float32)}
    for i, st in enumerate(structs):
        for fname, vol in st.features(ctxs[i]).items():
            vol = np.asarray(vol, np.float32)
            features[fname] = vol if fname not in features else np.maximum(features[fname], vol)

    # the ledger, as realised
    tissue_i = []
    for i, st in enumerate(structs):
        terr = labels == i
        tis = terr & ~interior_i[i]
        tissue_i.append(tis)
        report["targets"][names[i]] = float(st.FRAC.v) * cover
        report["realised"][names[i]] = float(tis.sum()) / n_tot
        report["territory"][names[i]] = float(terr.sum()) / n_tot
        report["levels"][names[i]] = float(w_i[i])
        report["count"][names[i]] = ks[i]
        rad = (float(terr.sum()) / max(ks[i], 1) * 3.0 / (4.0 * np.pi)) ** (1 / 3)
        report["size_um"][names[i]] = (float(st.SIZE_UM.v), rad * float(um_per_vox))
        # nothing bites a structure any more: the pressure balance shares an interface
        # instead of carving, so this is 1.0 in BOTH declaration orders
        report["body_kept"][names[i]] = 1.0
    report["realised"]["lumen"] = float(interior.sum()) / n_tot
    bg = support & ~(labels >= 0)
    report["realised"]["background"] = float(bg.sum()) / n_tot
    report["targets"]["background"] = cover * max(1.0 - frac_sum, 0.0)
    report["cover"] = float(support.sum()) / n_tot
    report["iters"], report["converged"] = int(iters), bool(converged)

    R = np.stack(resp)
    field = {
             "Q6": Q6,
             "m": m,
             "resp": (R / (R.sum(0, keepdims=True) + 1e-12)).astype(np.float32),
             "dist": np.stack([np.where(labels == i, dists[i] - w_i[i], np.float32(1e3))
                               for i in range(len(structs))]).astype(np.float32),
             "terr": np.stack([labels == i for i in range(len(structs))]),
             "walls": np.array([-(1.0 - walls[i]) * w_i[i] for i in range(len(structs))]),
             "features": features, "psi": dist_bg.astype(np.float32), "thr": float(margin),
             "labels": labels, "score": score, "lumen": interior,
             "term_names": ["flow", "margin"] + names,
             "struct_names": names,
    "structures": recs,
    "u": u}
    return support, field, report
