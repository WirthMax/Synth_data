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
    pi_fixed = np.zeros((N_CONTEXT, K), np.float32)
    for c, fc in enumerate(_context_remap(b["th_contexts"])):
        pi_fixed[fc] = np.asarray(b["th_pi"])[c]
    pi = _t(pi_fixed[:, pidx])                               # [C, J]
    context_cells = _t(b["context_cells"])                   # [C]

    # --- pure stride-4 labels for the discriminative loss (present types only) ---
    pure = np.asarray(b["pure4"]).reshape(N4).astype(np.int64)   # 0 bg, 1..K type, -1 not pure
    type4 = pure - 1                                         # -> 0..K-1 type, -1 for bg/not-pure
    type4[~np.isin(type4, pidx)] = -1
    type4_pure = _t(type4, torch.long)

    type_t = dict(
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

    # --- global (padded to n_global with a mask; variable ARCH length) + per-marker ---
    gu = np.asarray(b["th_global_u"], np.float32)
    g_pad = np.full(n_global, 0.5, np.float32)
    g_mask = np.zeros(n_global, np.float32)
    n = min(len(gu), n_global)
    g_pad[:n], g_mask[:n] = gu[:n], 1.0
    glob = dict(u=_t(g_pad), mask=_t(g_mask))
    marker = dict(noise_u=_t(b["th_noise4"]), mu_u=_t(b["th_mu_u"]),
                  offset=_t(b["th_channel_offsets"]), present=torch.ones(M, dtype=torch.bool))

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

    def _stack(key):
        return torch.stack([s["dense"][key] for s in samples])

    dense = {k: _stack(k) for k in ("nucleus", "cell", "flow", "nuc_flow", "ignore",
                                    "cell_visible", "nuc_visible", "structure", "structure_mask")}
    glob = dict(u=torch.stack([s["glob"]["u"] for s in samples]),
                mask=torch.stack([s["glob"]["mask"] for s in samples]))
    marker = dict(noise_u=noise, mu_u=mu_u, offset=offset, present=present)
    tgt = dict(type=[s["type"] for s in samples], dense=dense, marker=marker,
               strength=strength, **{"global": glob})
    return tile, present, tgt
