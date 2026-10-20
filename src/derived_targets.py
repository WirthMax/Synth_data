"""Derived training targets from the exact labels

These are deterministic functions of the exported label arrays (inst / type_map / context_map) --
they are the things the model is actually trained against, and are meant to be computed in the data
loader (optionally cached). Nothing here involves a segmentation network: the Omnipose-style flows
are a formula applied to the generator's own masks.

All strides divide the tile size (160 -> 40 at stride 4, 10 at stride 16).
"""
import numpy as np
from scipy import ndimage as ndi

import config as cfg


def cell_mask(inst, visible_ids):
    """Boolean foreground of VISIBLE cells: (inst > 0) & (inst is a visible id)."""
    vis = np.asarray(list(visible_ids), dtype=inst.dtype)
    return np.isin(inst, vis) & (inst > 0) if vis.size else np.zeros(inst.shape, bool)


def instance_flows(inst, visible_ids, smooth=1.0):
    """Omnipose-style flow field: the normalised gradient of each visible instance's distance
    field. Returns float32 [2, H, W] = (flow_y, flow_x), zero outside visible cells.

    Per instance: crop to its bbox (padded 2 px), distance-transform, lightly smooth (limits the
    pixelation ridges a raw EDT leaves), take the gradient and normalise to unit length. This is a
    pure encoding of the generator's exact mask -- no Cellpose/Omnipose model is used.
    """
    H, W = inst.shape
    flow = np.zeros((2, H, W), np.float32)
    vis = set(int(v) for v in visible_ids)
    objs = ndi.find_objects(inst.astype(np.int64))
    for i, sl in enumerate(objs, start=1):
        if sl is None or i not in vis:
            continue
        # pad the bbox by 2 px (clipped to the frame) so the EDT sees the true boundary
        ys, xs = sl
        y0, y1 = max(ys.start - 2, 0), min(ys.stop + 2, H)
        x0, x1 = max(xs.start - 2, 0), min(xs.stop + 2, W)
        m = inst[y0:y1, x0:x1] == i
        if not m.any():
            continue
        d = ndi.distance_transform_edt(m)
        d = ndi.gaussian_filter(d, smooth) * m
        gy, gx = np.gradient(d)
        n = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
        fy, fx = gy / n, gx / n
        flow[0, y0:y1, x0:x1][m] = fy[m]
        flow[1, y0:y1, x0:x1][m] = fx[m]
    return flow


def fraction_map(label_map, n_classes, stride, ignore=None):
    """Per stride x stride block, the fraction of (non-ignore) pixels in each class 0..n_classes-1.

    Returns (frac [n_classes, H//s, W//s] float32, valid [H//s, W//s] bool). `ignore` pixels are
    removed from the denominator; a block that is entirely ignore is marked not-valid (and its
    fractions are 0). Within a valid block the fractions sum to 1 (every non-ignore label must fall
    in 0..n_classes-1).
    """
    H, W = label_map.shape
    Hs, Ws = H // stride, W // stride
    lm = label_map[:Hs * stride, :Ws * stride]
    blocks = lm.reshape(Hs, stride, Ws, stride).transpose(0, 2, 1, 3).reshape(Hs, Ws, stride * stride)
    keep = np.ones_like(blocks, bool) if ignore is None else (blocks != ignore)
    denom = keep.sum(-1).astype(np.float32)
    valid = denom > 0
    frac = np.zeros((n_classes, Hs, Ws), np.float32)
    for c in range(n_classes):
        frac[c] = ((blocks == c) & keep).sum(-1) / np.maximum(denom, 1.0)
    return frac, valid


def type_fractions(type_map, k_true, stride, ignore=cfg.IGNORE_LABEL):
    """Type fraction map [1 + k_true, H//s, W//s] (class 0 = background), ignore removed."""
    return fraction_map(type_map, 1 + k_true, stride, ignore=ignore)


def context_fractions(context_map, n_contexts, stride):
    """Context fraction map [n_contexts, H//s, W//s] (class 0 = background tissue)."""
    return fraction_map(context_map, n_contexts, stride, ignore=None)


def pure_locations(type_map, k_true, stride=4, thresh=0.8, ignore=cfg.IGNORE_LABEL):
    """Stride-4 blocks dominated by a single class: returns an int map [H//s, W//s] holding the
    dominant class (0 = background, 1..k_true = type) where it covers >= thresh of the block's
    non-ignore pixels, else -1 (not pure, incl. all-ignore blocks). Seeds and the discriminative
    loss filter this to type > 0."""
    frac, valid = type_fractions(type_map, k_true, stride, ignore=ignore)
    dom = frac.argmax(0)
    dom_frac = frac.max(0)
    pure = np.where(valid & (dom_frac >= thresh), dom, -1).astype(np.int32)
    return pure
