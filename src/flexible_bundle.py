"""Generation-side bundling for flexible-mode training data (plan Phase 6.1).

One flexible sample is more than (image, theta): the v3 losses need dense truth (cell/nucleus masks,
Omnipose flows), stride-4/16 type and context fraction maps, and pure-location labels. Those derived
targets use scipy (distance transforms, labelling), so they are computed HERE, on the generation
(synmac) side, and stored -- the synmodel loader then only reshapes/pads/normalises (no scipy), per
the plan's "derived targets computed in the loader, optionally cached".

`build_bundle` returns a flat dict of numpy arrays ready for np.savez; `targets_v3.build_sample_targets`
(synmodel side) consumes exactly these keys. Keeping the two in one documented contract is the whole
point of splitting here.
"""
import numpy as np

import config as cfg
import derived_targets as dt
import flexible_types as ft
import gen_dataset as gd


def _visible_ids(cells):
    return [int(r["id"]) for r in cells if bool(r["visible"])]


def _context_cells(cells, n_context):
    """Visible-cell count per context (0 = background, 1.. = CONTEXT_KINDS), the gate the context
    Dirichlet loss uses (plan 5.3c needs >= min_context_cells)."""
    out = np.zeros(n_context, np.int64)
    for r in cells:
        if bool(r["visible"]) and 0 <= int(r["context"]) < n_context:
            out[int(r["context"])] += 1
    return out


def build_bundle(i, seed, curriculum=None, with_artifacts=False, pool_bank=None):
    """Render one flexible sample and compute every array the v3 target-builder needs.

    Returns a flat dict: the image, the theta v3 arrays (prefixed `th_`), the raw 2D labels, the
    precomputed derived targets, and per-context cell counts. K here is the sample's realised type
    count; the loader pads to K_max.
    """
    pool_bank = cfg.POOL_BANK if pool_bank is None else pool_bank
    _, img, theta, mask, geojson, cyc, labels = gd.draw_flexible(
        i, seed=seed, curriculum=curriculum, with_artifacts=with_artifacts, pool_bank=pool_bank)

    K = int(theta["K"])
    n_context = 1 + len(cfg.CONTEXT_KINDS)
    type_map = labels["type_map"]
    context_map = labels["context_map"]
    inst = labels["inst"]
    vis = _visible_ids(labels["cells"])

    frac4, valid4 = dt.type_fractions(type_map, K, stride=4)        # [1+K, h4, w4]
    frac16, _ = dt.type_fractions(type_map, K, stride=16)           # [1+K, h16, w16]
    cfrac16, _ = dt.context_fractions(context_map, n_context, stride=16)
    pure4 = dt.pure_locations(type_map, K, stride=4)                # [h4, w4], 0 bg / 1..K / -1
    cellmask = dt.cell_mask(inst, vis)
    flow = dt.instance_flows(inst, vis)                             # [2, H, W]

    bundle = dict(
        image=img.astype(np.uint16),
        type_map=type_map.astype(np.uint8),
        nuc=labels["nuc"].astype(np.uint8),
        cell_mask=cellmask.astype(np.bool_),
        flow=flow.astype(np.float32),
        frac4=frac4.astype(np.float32), valid4=valid4.astype(np.bool_),
        frac16=frac16.astype(np.float32),
        cfrac16=cfrac16.astype(np.float32),
        pure4=pure4.astype(np.int32),
        present=np.asarray(labels["present"], np.bool_),           # frame-present theta-types [K]
        context_cells=_context_cells(labels["cells"], n_context),
    )
    bundle.update({f"th_{k}": np.asarray(v) for k, v in theta.items()})
    return bundle
