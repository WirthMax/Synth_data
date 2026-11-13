"""Flow field -> instance labels, and the instance-level scores that judge it.

The dense heads predict an Omnipose-style flow: the unit gradient of each object's distance
transform. Its sinks are NOT points -- every pixel of a cell flows onto the cell's medial ridge, a
line for an elongated cell -- so the decoder must treat a ridge as ONE sink, not cut it wherever the
endpoint density dips. The old notebook decoder thresholded a blurred endpoint histogram at a fixed
fraction of its GLOBAL maximum, so a long ridge (endpoints spread thin) fell apart into several
basins while a round cell (endpoints piled on one pixel) set the maximum: measured on held-out
synthetic tiles it split 44 ground-truth cells into 60 even when handed the ground-truth flows.

Here the endpoints are rasterised, dilated by `merge_radius` px and connected-component labelled,
so a ridge is one component however thin the endpoints lie along it; two neighbouring cells stay
apart as long as their ridges are more than 2 x merge_radius apart. The integration is deliberately
SHORT (n_iter ~ a cell half-width): run to convergence, the points keep sliding ALONG the ridge into
its separate local maxima and the long cell splits again -- measured on 8 held-out tiles, 37% of
cells with elongation > 2.5 split at 200 steps. GT-flow ceiling (F1 @ IoU 0.5, 214 cells / 178
nuclei): 0.79 / 0.87 with the old decoder, 0.91 / 0.97 here (merge_radius 2, n_iter 12). Fragments below `min_size` are
absorbed into the neighbour they share the longest border with rather than deleted, so an
over-split cell heals instead of losing a piece.

Numpy + torch only -- no scipy -- so the same decoder runs in the synmodel training env (the
rounds evaluation) and in the synmac notebooks. Connected components are a masked max-pool label
propagation, which also runs on the GPU.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _as_t(a, device, dtype=torch.float32):
    return a.to(device=device, dtype=dtype) if torch.is_tensor(a) else torch.as_tensor(
        np.asarray(a), device=device, dtype=dtype)


def follow_flows(flow, fg, n_iter=12, step=1.0, eps=1e-3, device="cpu"):
    """Euler-integrate every foreground pixel along the flow. The flow is zeroed outside the
    foreground (a pixel cannot be pulled through background into the next cell) and normalised to
    unit length wherever it is not ~0 -- the network's tanh output shrinks near ridges, and an
    un-normalised step would stall pixels half way and spawn a sink in the middle of a cell.
    Returns (endpoints [N, 2] float (y, x), start pixels [N, 2] long)."""
    f = _as_t(flow, device)                                        # [2, H, W] = (fy, fx)
    m = _as_t(fg, device, torch.bool)
    H, W = m.shape
    f = f * m
    n = f.norm(dim=0, keepdim=True)
    f = torch.where(n > eps, f / n.clamp(min=eps), torch.zeros_like(f))
    start = torch.nonzero(m)                                       # [N, 2] (y, x)
    if start.numel() == 0:
        return start.float(), start
    p = start.float()
    grid_f = f[None]                                               # [1, 2, H, W]
    sy, sx = 2.0 / max(H - 1, 1), 2.0 / max(W - 1, 1)
    for _ in range(int(n_iter)):
        # grid_sample wants (x, y) in [-1, 1]; align_corners=True maps -1/+1 onto pixel centres
        g = torch.stack([p[:, 1] * sx - 1.0, p[:, 0] * sy - 1.0], -1)[None, None]
        v = F.grid_sample(grid_f, g, mode="bilinear", align_corners=True)[0, :, 0].T  # [N, 2]
        p = p + step * v
        p[:, 0].clamp_(0, H - 1)
        p[:, 1].clamp_(0, W - 1)
    return p, start


def _cc_label(mask):
    """4/8-connected component labels of a bool tensor by max-pool label propagation (no scipy).
    Each component ends up carrying the largest flat index inside it; relabelled to 1..n."""
    H, W = mask.shape
    lab = torch.where(mask, torch.arange(1, H * W + 1, device=mask.device,
                                         dtype=torch.float32).view(H, W), torch.zeros(H, W, device=mask.device))
    lab = lab[None, None]
    m = mask[None, None]
    while True:
        # several steps per convergence check -- the check is the expensive sync on a GPU
        prev = lab
        for _ in range(16):
            lab = torch.where(m, F.max_pool2d(lab, 3, stride=1, padding=1), lab)
        if torch.equal(lab, prev):
            break
    lab = lab[0, 0].long()
    _, inv = torch.unique(lab, return_inverse=True)
    return inv if bool((lab == 0).any()) else inv + 1               # background stays 0


def cluster_sinks(endpoints, start, shape, merge_radius=2):
    """Endpoints -> instance labels: rasterise, dilate by merge_radius (a square, so a diagonal
    ridge stays connected), connected-component label, and give every start pixel the label of the
    component its endpoint landed in."""
    H, W = shape
    inst = torch.zeros(H, W, dtype=torch.long, device=start.device)
    if start.numel() == 0:
        return inst
    e = endpoints.round().long()
    hit = torch.zeros(1, 1, H, W, device=start.device)
    hit[0, 0, e[:, 0], e[:, 1]] = 1.0
    k = 2 * int(merge_radius) + 1
    blob = F.max_pool2d(hit, k, stride=1, padding=int(merge_radius))[0, 0] > 0
    comp = _cc_label(blob)
    inst[start[:, 0], start[:, 1]] = comp[e[:, 0], e[:, 1]]
    return inst


def relabel(inst):
    """Labels -> 0..n consecutive, background kept at 0. Numpy in, numpy out."""
    inst = np.asarray(inst)
    u, inv = np.unique(inst, return_inverse=True)
    out = inv.reshape(inst.shape)
    return out if u[0] == 0 else out + 1


def merge_small(inst, min_size=15):
    """Absorb every fragment below min_size px into the touching instance it shares the longest
    border with (4-neighbourhood); a fragment touching nothing but background is dropped. Smallest
    first, so a chain of slivers collapses into the cell they were cut from."""
    inst = relabel(inst).astype(np.int64)
    while True:
        n = int(inst.max())
        if n == 0:
            return inst
        size = np.bincount(inst.ravel(), minlength=n + 1)
        small = [i for i in np.argsort(size) if i > 0 and 0 < size[i] < min_size]
        if not small:
            return relabel(inst)
        i = small[0]
        pairs = []
        for a, b in ((inst[:, :-1], inst[:, 1:]), (inst[:-1, :], inst[1:, :])):
            s = (a == i) & (b != i) & (b > 0)
            pairs.append(b[s])
            s = (b == i) & (a != i) & (a > 0)
            pairs.append(a[s])
        nb = np.concatenate(pairs)
        inst[inst == i] = np.bincount(nb).argmax() if nb.size else 0
        inst = relabel(inst)


def flow_error_drop(inst, flow, flow_threshold=0.4):
    """Cellpose's consistency check: recompute the flow each decoded mask WOULD have and drop masks
    whose mean squared disagreement with the predicted flow exceeds flow_threshold. Raising the
    threshold is "more lenient" -- it KEEPS more masks; it never merges fragments, so it is no cure
    for over-segmentation, only for false positives. Needs scipy (derived_targets), so synmac only."""
    from derived_targets import instance_flows
    inst = relabel(inst)
    ids = np.unique(inst)[1:]
    ref = instance_flows(inst, ids)
    err = ((np.asarray(flow) - ref) ** 2).sum(0)
    n = int(inst.max())
    mse = np.bincount(inst.ravel(), weights=err.ravel(), minlength=n + 1) / (
        np.bincount(inst.ravel(), minlength=n + 1) + 1e-12)
    keep = mse <= flow_threshold
    keep[0] = True
    return relabel(np.where(keep[inst], inst, 0))


def flow_to_instances(flow, fg, merge_radius=2, min_size=15, n_iter=12, step=1.0,
                      flow_threshold=None, device="cpu"):
    """The whole decode: follow -> cluster -> heal small fragments [-> optional flow-error drop].
    `flow` is [2, H, W] (fy, fx), `fg` a bool [H, W] foreground. Returns int32 [H, W] labels."""
    fg_t = _as_t(fg, device, torch.bool)
    end, start = follow_flows(flow, fg_t, n_iter=n_iter, step=step, device=device)
    inst = cluster_sinks(end, start, tuple(fg_t.shape), merge_radius=merge_radius).cpu().numpy()
    inst = merge_small(inst, min_size=min_size)
    if flow_threshold is not None:
        inst = flow_error_drop(inst, np.asarray(flow.cpu() if torch.is_tensor(flow) else flow),
                               flow_threshold=flow_threshold)
    return inst.astype(np.int32)


def match(gt, pred, thr=0.5):
    """Instance scores at IoU > thr. Above 0.5 a match is unique, so counting the pairs over the
    threshold IS the optimal assignment -- no Hungarian needed. Returns precision / recall / F1, the
    mean IoU of the matched pairs and both counts."""
    gt, pred = relabel(gt), relabel(pred)
    ng, npd = int(gt.max()), int(pred.max())
    C = np.bincount(gt.ravel().astype(np.int64) * (npd + 1) + pred.ravel(),
                    minlength=(ng + 1) * (npd + 1)).reshape(ng + 1, npd + 1)
    iou = C / (C.sum(1, keepdims=True) + C.sum(0, keepdims=True) - C + 1e-12)
    iou = iou[1:, 1:]
    hit = iou > thr
    tp = int(hit.sum())
    return dict(n_gt=ng, n_pred=npd, prec=tp / max(npd, 1), rec=tp / max(ng, 1),
                f1=2 * tp / max(ng + npd, 1), miou=float(iou[hit].mean()) if tp else 0.0)
