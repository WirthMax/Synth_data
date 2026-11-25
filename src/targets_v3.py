"""Synmodel-side target builder for the v3 losses (plan Phase 6.1).

Turns a stored `flexible_bundle.build_bundle` dict (numpy arrays: image, theta v3, raw labels and the
scipy-computed derived targets) into the per-sample target dict `losses_v3.total_loss` consumes. Pure
numpy + torch, NO scipy, so it runs at data-load time in the synmodel env. The matching contract is
documented by `losses_v3` and exercised against a REAL generated bundle in the Phase 6 tests.

`collate_v3` batches samples of different M (channels) and J (present types): M is padded with a
`present` mask, the type targets stay ragged per sample (the loss loops over samples), and C (the
fixed context count) never varies. K here is K_slots, the model's slot count.
"""
import numpy as np
import torch

import config as cfg

N_CONTEXT = 1 + len(cfg.CONTEXT_KINDS)


def _t(a, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype)


def _context_remap(contexts):
    """sample context names -> fixed context index (background 0, then CONTEXT_KINDS). Returns a
    list the same length as `contexts`."""
    out = []
    for nm in contexts:
        nm = nm.decode() if isinstance(nm, bytes) else str(nm)
        out.append(0 if nm == "background" else cfg.CONTEXT_KINDS.index(nm) + 1)
    return out


# bundle arrays by spatial layout, for the label-consistent flip / rot90 below
_HW_FIRST = ("image",)                                       # [H, W, ...]
_HW_LAST = ("type_map", "nuc", "cell_mask", "inst", "valid4", "pure4",
            "frac4", "frac16", "cfrac16")                    # [..., H, W]
_VEC_FIELDS = ("flow", "nuc_flow")                           # [2 = (dy, dx), H, W]


def _rot_vec(dy, dx, k_rot, flip):
    """(dy, dx) under a flip over W followed by k_rot ccw quarter turns: (dy, dx) -> (-dx, dy) per
    turn -- the same convention as losses_v3.augment_geometric."""
    if flip:
        dx = -dx
    for _ in range(k_rot % 4):
        dy, dx = -dx, dy
    return dy, dx


def augment_bundle(b, k_rot, flip):
    """A stored bundle flipped over W and turned k_rot x 90 deg counter-clockwise, labels and all.
    Done on the BUNDLE, before any target is derived, so every map the loss sees (instance labels,
    stride-4/16 fractions, the pure-location map, ignore) is transformed by construction, and the
    vector-valued fields -- flows and the per-channel registration offsets, both (dy, dx) -- are
    rotated as vectors. Global theta has no direction-valued entry (the ARCH.* scalars are
    rotation-invariant), so it is untouched. Intensities are never altered: the expression targets
    stay exact. Without this every training structure ran along +x (DIR_DEG is not sampled)."""
    k_rot = int(k_rot) % 4
    if k_rot == 0 and not flip:
        return b
    out = dict(b)

    def _sp(x, a0, a1):
        if flip:
            x = np.flip(x, axis=a1)
        return np.ascontiguousarray(np.rot90(x, k_rot, axes=(a0, a1)))

    for k in _HW_FIRST:
        if k in out:
            out[k] = _sp(np.asarray(out[k]), 0, 1)
    for k in _HW_LAST:
        if k in out:
            x = np.asarray(out[k])
            out[k] = _sp(x, x.ndim - 2, x.ndim - 1)
    for k in _VEC_FIELDS:
        if k in out:
            f = _sp(np.asarray(out[k]), 1, 2)
            dy, dx = _rot_vec(f[0], f[1], k_rot, flip)
            out[k] = np.stack([dy, dx]).astype(f.dtype)
    if "th_channel_offsets" in out:
        o = np.asarray(out["th_channel_offsets"])
        dy, dx = _rot_vec(o[..., 0], o[..., 1], k_rot, flip)
        out["th_channel_offsets"] = np.stack([dy, dx], -1).astype(o.dtype)
    # a structure's declared direction IS in theta since 2026-11-16 (it is sampled now): DIR_DEG is
    # a headless in-plane axis normalised over [-90, 90], so a flip mirrors it (u -> 1 - u) and a
    # quarter turn shifts it by half its range, wrapping
    if "th_global_names" in out and "th_global_u" in out:
        gu = np.array(out["th_global_u"], np.float64)
        isdir = np.array([str(n).endswith(".DIR_DEG") for n in np.asarray(out["th_global_names"])])
        if isdir.any():
            u = gu[isdir]
            if flip:
                u = 1.0 - u
            if k_rot % 2:
                u = (u + 0.5) % 1.0
            gu[isdir] = np.clip(u, 1e-4, 1.0 - 1e-4)
            out["th_global_u"] = gu
    return out


# bundle arrays indexed by image channel, for drop_channels: (key, channel axis)
_CH_AXES = (("image", -1), ("th_noise4", 0), ("th_mu_u", 0), ("th_positive", 1),
            ("th_level_u", 1), ("th_channel_order", 0), ("th_channel_offsets", 0))


def drop_channels(b, keep):
    """A stored bundle restricted to the image channels `keep` (bool [M]), with every per-channel
    target cut the same way -- the tile as it would be had those antibodies not been in the panel.
    The scene is untouched, so every label stays exact; the counterstain is never dropped (the
    caller keeps it). Two types that differed only in a dropped marker become twins with separate
    territories, which is honest: that is what the smaller panel shows.

    Why: in a synthetic tile every channel is rendered from the same cells, so a boundary or a
    nucleus seen in one channel is confirmed by a dozen others, and the model can learn to trust the
    CONSENSUS instead of any single stain. Real panels do not offer that redundancy. Seeing each
    scene under several random sub-panels takes it away."""
    keep = np.asarray(keep, bool)
    if keep.all():
        return b
    out = dict(b)
    for key, ax in _CH_AXES:
        if key in out:
            out[key] = np.compress(keep, np.asarray(out[key]), axis=ax)
    out["th_M"] = np.int64(int(keep.sum()))
    return out


def build_sample_targets(b, k_slots, n_global):
    """One sample's target dict. `b` is the stored bundle (keys from flexible_bundle.build_bundle),
    `k_slots` the model's slot count, `n_global` the model's padded global length."""
    K = int(b["th_K"])
    M = int(b["th_M"])
    present = np.asarray(b["present"], bool)                  # frame-present theta-types [K]
    pidx = np.flatnonzero(present)
    J = pidx.size
    frac4 = np.asarray(b["frac4"])                            # [1+K, h4, w4]
    frac16 = np.asarray(b["frac16"])                          # [1+K, h16, w16]
    h4, w4 = frac4.shape[1:]
    h16, w16 = frac16.shape[1:]
    N4, N16 = h4 * w4, h16 * w16

    # --- type-level truth, restricted to present types ---
    Mgt = _t(frac4[1 + pidx].reshape(J, N4))                 # [J, N4]
    frac16_t = _t(np.concatenate([frac16[:1], frac16[1 + pidx]], 0).reshape(1 + J, N16))
    token_mask = _t(frac16.sum(0).reshape(N16) > 0.5)        # valid (not all-ignore) stride-16 blocks

    positive = _t(np.asarray(b["th_positive"])[pidx])        # [J, M]
    level = _t(np.asarray(b["th_level_u"])[pidx])            # [J, M] normalised, 0 where not positive
    geometry = _t(np.asarray(b["th_geometry_u"])[pidx])      # [J, 5]
    E_gt = torch.where(positive > 0.5, level, torch.zeros_like(level))

    # --- compositions: remap sample contexts to the fixed C, select present type columns ---
    # the composition target is what the FRAME shows (flexible_bundle._realised_composition, already
    # in the fixed context order); bundles from before 2026-11-16 only carry theta's pi
    pi_fixed = np.zeros((N_CONTEXT, K), np.float32)
    if "comp_real" in b:
        pi_fixed[:, :] = np.asarray(b["comp_real"], np.float32)[:N_CONTEXT, :K]
    else:
        for c, fc in enumerate(_context_remap(b["th_contexts"])):
            pi_fixed[fc] = np.asarray(b["th_pi"])[c]
    pi = _t(pi_fixed[:, pidx])                               # [C, J]
    context_cells = _t(b["context_cells"])                   # [C]

    # --- pure stride-4 labels for the discriminative loss (present types only) ---
    pure = np.asarray(b["pure4"]).reshape(N4).astype(np.int64)   # 0 bg, 1..K type, -1 not pure
    type4 = pure - 1                                         # -> 0..K-1 type, -1 for bg/not-pure
    type4[~np.isin(type4, pidx)] = -1
    type4_pure = _t(type4, torch.long)

    # per-instance truth for the query head: present-type index j (-1 = not a present type) and context
    inst_t = {}
    if "inst_type" in b:
        j_of = np.full(max(K, 1), -1, np.int64)
        j_of[pidx] = np.arange(J)
        it = np.asarray(b["inst_type"]).astype(np.int64)
        inst_t = dict(inst_j=_t(np.where(it >= 0, j_of[np.clip(it, 0, max(K, 1) - 1)], -1), torch.long),
                      inst_ctx=_t(np.asarray(b["inst_ctx"]).astype(np.int64), torch.long))

    type_t = dict(**inst_t,
        Mgt=Mgt, E_gt=E_gt, mk=torch.ones(M), positive=positive, level=level, geometry=geometry,
        pi=pi, context_cells=context_cells, frac16=frac16_t, token_mask=token_mask,
        type4_pure=type4_pure, seed_frac=torch.zeros(k_slots, J))  # seed_frac filled at train time

    # --- dense targets ---
    type_map = np.asarray(b["type_map"])
    ignore = type_map == cfg.IGNORE_LABEL
    dense = dict(
        nucleus=_t(b["nuc"]).float(), cell=_t(b["cell_mask"]).float(),
        flow=_t(b["flow"]), nuc_flow=_t(b["nuc_flow"]), ignore=_t(ignore, torch.bool),
        cell_visible=_t(b["cell_mask"]).float(),
        nuc_visible=_t(np.asarray(b["nuc"]) > 0).float(),    # nucleus flow-loss region
        structure=_t(b["cfrac16"]),                          # [C, h16, w16] soft target
        structure_mask=torch.ones(h16, w16))
    if "inst" in b:                                          # bundles from before it was stored lack it
        dense["inst"] = _t(np.asarray(b["inst"]).astype(np.int64), torch.long)

    # --- global (padded to n_global with a mask; variable ARCH length) + per-marker ---
    gu = np.asarray(b["th_global_u"], np.float32)
    g_pad = np.full(n_global, 0.5, np.float32)
    g_mask = np.zeros(n_global, np.float32)
    n = min(len(gu), n_global)
    g_pad[:n], g_mask[:n] = gu[:n], 1.0
    glob = dict(u=_t(g_pad), mask=_t(g_mask))
    # the nuclear counterstain flag: canonical marker 0 is DAPI, and th_channel_order maps each
    # IMAGE channel to its canonical marker (the panel is permuted per tile)
    order = np.asarray(b["th_channel_order"]) if "th_channel_order" in b else np.arange(M)
    marker = dict(noise_u=_t(b["th_noise4"]), mu_u=_t(b["th_mu_u"]),
                  offset=_t(b["th_channel_offsets"]), present=torch.ones(M, dtype=torch.bool),
                  ref=_t((order == 0).astype(np.float32)))

    return dict(type=type_t, dense=dense, marker=marker, glob=glob,
                image=_t(b["image"]), M=M, K=K, J=J)


def seed_frac_from(seeds, Mgt, k_slots):
    """seed_frac[k, j] = GT fraction of present-type j in the stride-4 block at slot k's seed (plan
    5.2). seeds: [k_slots] long (stride-4 flat index); Mgt: [J, N4]. Slots past n_seeds get 0."""
    J, N4 = Mgt.shape
    out = Mgt.new_zeros(k_slots, J)
    valid = (seeds >= 0) & (seeds < N4)
    if valid.any():
        out[valid] = Mgt[:, seeds[valid]].T
    return out


def collate_v3(samples, m_max=None):
    """Batch samples with different M. Pads the image channels to m_max (present mask marks the
    real ones), per-marker-normalises each tile and carries the per-marker expression `strength`,
    stacks dense/global, and keeps the per-sample `type` dicts as a list (the loss loops over them).
    Returns (tile [B, m_max, H, W], present [B, m_max], tgt); `tgt['strength']` is [B, m_max]."""
    B = len(samples)
    m_max = m_max or max(int(s["M"]) for s in samples)
    H, W = samples[0]["image"].shape[:2]
    tile = torch.zeros(B, m_max, H, W)
    present = torch.zeros(B, m_max, dtype=torch.bool)
    strength = torch.zeros(B, m_max)
    noise = torch.zeros(B, m_max, samples[0]["marker"]["noise_u"].shape[-1])
    mu_u = torch.zeros(B, m_max)
    offset = torch.zeros(B, m_max, 2)
    ref = torch.zeros(B, m_max)
    from model import normalize_markers
    for b, s in enumerate(samples):
        M = int(s["M"])
        img = s["image"]                                     # [H, W, M] uint16 tensor
        norm, strn = normalize_markers(img.cpu().numpy())    # per-marker [0,1] + strength [M]
        tile[b, :M] = torch.as_tensor(norm).permute(2, 0, 1)
        strength[b, :M] = torch.as_tensor(strn)
        present[b, :M] = True
        noise[b, :M] = s["marker"]["noise_u"]
        mu_u[b, :M] = s["marker"]["mu_u"]
        offset[b, :M] = s["marker"]["offset"]
        if "ref" in s["marker"]:
            ref[b, :M] = s["marker"]["ref"]

    def _stack(key):
        return torch.stack([s["dense"][key] for s in samples])

    dense = {k: _stack(k) for k in ("nucleus", "cell", "flow", "nuc_flow", "ignore",
                                    "cell_visible", "nuc_visible", "structure", "structure_mask")}
    if all("inst" in s["dense"] for s in samples):
        dense["inst"] = _stack("inst")
    glob = dict(u=torch.stack([s["glob"]["u"] for s in samples]),
                mask=torch.stack([s["glob"]["mask"] for s in samples]))
    marker = dict(noise_u=noise, mu_u=mu_u, offset=offset, present=present)
    tgt = dict(type=[s["type"] for s in samples], dense=dense, marker=marker,
               strength=strength, ref=ref, **{"global": glob})
    return tile, present, tgt
