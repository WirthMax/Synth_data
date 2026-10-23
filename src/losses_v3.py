"""Inversion model v3 losses, matching and the anchored-discovery schedule (plan Phase 5).

The cell types are an unordered set recovered as slots (model_v3.TypeSlots), so every type term is
charged only after a per-sample Hungarian match of slots to the ground-truth types. The match uses
maps and means ONLY -- never the predicted variance -- so a vague slot cannot win by being equally
unsure about everything (plan 5.2); the variance is trained by the NLL *after* matching.

Runs in the synmodel (torch) env, which has NO scipy, so the assignment is a small self-contained
Hungarian (`linear_sum_assignment`) rather than `scipy.optimize`'s -- cross-checked against scipy
and brute force in the tests. Everything else is torch.

The loss functions are written to be testable in isolation on synthetic targets; wiring them to the
generator's real labels (the padded theta v3 target contract) is Phase 6. Target tensors follow the
padded (K slots, M markers, C contexts) convention; present/valid masks say which entries are real.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


# --- anchored-discovery schedule (plan 5.4) ---------------------------------------------------
def schedule(step, total_steps):
    """The teacher-seeding probability and seed-anchor weight as a function of training progress
    f = step/total_steps. Early on every present type is force-seeded (p_teacher=1) and matching
    follows the seeds (w_seed high) so assignments cannot flip; both decay away so inference-time
    FPS seeding takes over."""
    f = float(step) / max(float(total_steps), 1.0)
    if f < 0.10:
        p_teacher, w_seed = 1.0, 10.0
    elif f < 0.60:
        u = (f - 0.10) / 0.50                                  # 0..1 over [0.10, 0.60]
        p_teacher = 0.5 * (1.0 + np.cos(np.pi * u))            # cosine 1 -> 0
        w_seed = 10.0 * (1.0 - u)                              # linear 10 -> 0
    else:
        p_teacher, w_seed = 0.0, 0.0
    return dict(p_teacher=float(p_teacher), w_seed=float(w_seed), f=f)


# --- Hungarian assignment (no scipy) ----------------------------------------------------------
def _hungarian(a):
    """Min-cost assignment for a cost matrix with n rows <= m cols. Returns `col[i]`, the column
    matched to each row. Classic O(n^2 m) shortest-augmenting-path algorithm with potentials."""
    a = np.asarray(a, dtype=np.float64)
    n, m = a.shape
    assert n <= m, "internal: _hungarian needs rows <= cols"
    INF = np.inf
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)                            # p[j] = row matched to col j (1-based)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    col = np.full(n, -1, dtype=int)
    for j in range(1, m + 1):
        if p[j] > 0:
            col[p[j] - 1] = j - 1
    return col


def linear_sum_assignment(cost):
    """Drop-in for scipy.optimize.linear_sum_assignment: returns (row_ind, col_ind), the optimal
    assignment of min(R, C) pairs minimising total cost."""
    cost = np.asarray(cost, dtype=np.float64)
    R, C = cost.shape
    if R <= C:
        col = _hungarian(cost)
        return np.arange(R), col
    row = _hungarian(cost.T)                                  # cols become rows
    return row, np.arange(C)


# --- matching cost (Appendix A.6) -------------------------------------------------------------
@dataclass
class MatchWeights:
    dice: float = 5.0
    bce: float = 5.0
    expr: float = 1.0
    exist: float = 2.0
    seed: float = 10.0


@torch.no_grad()
def match_one(A, Mgt, E_pred, E_gt, mk, p_exist, seed_frac, w: MatchWeights):
    """A: [K, N] type-slot maps; Mgt: [J, N] true fractions; E_pred: [K, M]; E_gt: [J, M];
    mk: [M] present-marker mask (float); p_exist: [K]; seed_frac: [K, J]. Returns (slot_idx,
    type_idx) long tensors of length J (every present type matched to a distinct slot). NO variance
    enters the cost (plan 5.2)."""
    K, N = A.shape
    J = Mgt.shape[0]
    assert J <= K, f"J={J} present types but only K={K} slots"
    A = A.float()
    Mgt = Mgt.float()
    inter = A @ Mgt.T
    dice = 1 - (2 * inter + 1) / (A.sum(1)[:, None] + Mgt.sum(1)[None] + 1)
    la, l1a = A.clamp(1e-6).log(), (1 - A).clamp(1e-6).log()
    bce = -(la @ Mgt.T + l1a @ (1 - Mgt).T) / N
    expr = ((E_pred[:, None] - E_gt[None]).abs() * mk).sum(-1) / mk.sum().clamp(min=1)
    C = (w.dice * dice + w.bce * bce + w.expr * expr
         - w.exist * p_exist[:, None] - w.seed * seed_frac)
    r, c = linear_sum_assignment(C.detach().cpu().numpy())
    return (torch.as_tensor(r, device=A.device, dtype=torch.long),
            torch.as_tensor(c, device=A.device, dtype=torch.long))


def dirichlet_nll(alpha, target):
    """alpha, target: [J] for one context (J >= 2). Negative log-likelihood of `target` (a
    composition over the present types) under a Dirichlet(alpha)."""
    t = target.clamp(min=1e-3)
    t = t / t.sum()
    return (torch.lgamma(alpha).sum() - torch.lgamma(alpha.sum())
            - ((alpha - 1) * t.log()).sum())


# --- discriminative embedding loss (Appendix A.7) ---------------------------------------------
def discriminative_loss(e, y, dv=0.5, dd=1.5, gamma=1e-3):
    """e: [n, d] embeddings at pure locations of ONE sample; y: [n] type labels. A matching-free
    grouping signal: pull each type's embeddings within dv of their mean, push type means apart to
    2*dd. Gives seeds/slots a clustered space to latch onto from step 0 (plan 5.4)."""
    labels = y.unique()
    mus = torch.stack([e[y == l].mean(0) for l in labels])
    pull = torch.stack([F.relu((e[y == l] - mus[i]).norm(dim=1) - dv).pow(2).mean()
                        for i, l in enumerate(labels)]).mean()
    if len(labels) > 1:
        dist = torch.cdist(mus, mus)
        off = ~torch.eye(len(labels), dtype=torch.bool, device=e.device)
        push = F.relu(2 * dd - dist[off]).pow(2).mean()
    else:
        push = e.new_zeros(())
    return pull + push + gamma * mus.norm(dim=1).mean()


# --- token composition target (Appendix A.8) --------------------------------------------------
def token_comp_target(frac16, slot_of_type, K):
    """frac16: [1+J, N16] GT stride-16 fractions (row 0 = background); slot_of_type: [J] matched
    slot per type. Returns the target in slot order [K+1, N16] (background slot last)."""
    T = frac16.new_zeros(K + 1, frac16.shape[1])
    T[K] = frac16[0]
    T[slot_of_type] = frac16[1:]
    return T


# --- small NLL / dense helpers ----------------------------------------------------------------
_LOG2PI = float(np.log(2 * np.pi))

# upweight the marker localisation (mu) point-estimate loss so it is not drowned by the other
# per-marker / global terms while we check whether mu is learnable at all.
MU_LOSS_WEIGHT = 3.0


def gaussian_nll(mu, logvar, target):
    """Elementwise Gaussian negative log-likelihood (up to the constant kept for interpretability)."""
    return 0.5 * (logvar + (target - mu) ** 2 * torch.exp(-logvar) + _LOG2PI)


def logit(u, eps=1e-4):
    u = u.clamp(eps, 1 - eps)
    return torch.log(u / (1 - u))


def dice_bce(logits, target, mask=None, eps=1.0):
    """Dice + BCE for a soft/binary target. `mask` (same shape, float/bool) excludes ignore
    locations from both terms."""
    p = torch.sigmoid(logits)
    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * mask).sum() / mask.sum().clamp(min=1)
    inter = (p * target * mask).sum()
    denom = (p * mask).sum() + (target * mask).sum()
    dice = 1 - (2 * inter + eps) / (denom + eps)
    return bce + dice


def dice_bce_prob(p, target, eps=1.0):
    """Dice + BCE where the prediction is already a probability (e.g. a slot attention map), not a
    logit -- avoids a prob->logit->sigmoid round-trip that would blow up near 0/1."""
    p = p.clamp(1e-6, 1 - 1e-6)
    bce = F.binary_cross_entropy(p, target, reduction="mean")
    inter = (p * target).sum()
    dice = 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)
    return bce + dice


# --- per-sample type losses (a, b, c, d, e) ---------------------------------------------------
@dataclass
class LossWeights:
    exist: float = 1.0
    type_param: float = 1.0
    context: float = 1.0
    token_comp: float = 2.0
    matched_map: float = 2.0
    discrim: float = 1.0
    dense: float = 1.0
    global_marker: float = 1.0


def _expr_pred(expr):
    """E_pred[k,m] = sigmoid(pos_logit) * sigmoid(level_mu), the predicted mean normalised
    expression (plan 5.1)."""
    return torch.sigmoid(expr["positive"]) * torch.sigmoid(expr["level_mu"])


def type_losses_one(slot_out, tgt, w_match: MatchWeights, w: LossWeights,
                    exist_target="one", min_context_cells=10):
    """All matched type terms (a,b,c,d,e) for ONE sample at ONE slot-attention iteration.

    slot_out: the model's per-iteration dict (existence/expression/geometry/alpha/token_comp/maps)
      sliced to one sample, i.e. tensors without the batch axis.
    tgt: the per-sample target dict (see the test for the contract). Returns (dict of terms, match).
    """
    maps = slot_out["maps"]                                   # [K+1, N4]
    K = maps.shape[0] - 1
    A = maps[:K]                                              # type-slot maps [K, N4]
    Mgt = tgt["Mgt"]                                          # [J, N4]
    J = Mgt.shape[0]
    # slot_out may carry m_max (padded) markers; the sample's real markers are the first M, matching
    # the target. Slice the expression to them so a padded batch and a single sample agree.
    M = tgt["mk"].shape[0]
    expr = {k: v[:, :M] for k, v in slot_out["expression"].items()}
    E_pred = _expr_pred(expr)                                # [K, M]
    p_exist = torch.sigmoid(slot_out["existence"])           # [K]

    slot_idx, type_idx = match_one(A, Mgt, E_pred, tgt["E_gt"], tgt["mk"], p_exist,
                                   tgt["seed_frac"], w_match)

    terms = {}

    # (a) existence
    matched = torch.zeros(K, dtype=torch.bool, device=A.device)
    matched[slot_idx] = True
    if exist_target == "dice":
        tgt_pos = torch.zeros(K, device=A.device)
        dice_j = 1 - (2 * (A[slot_idx] * Mgt[type_idx]).sum(1) + 1) / (
            A[slot_idx].sum(1) + Mgt[type_idx].sum(1) + 1)
        tgt_pos[slot_idx] = (1 - dice_j)
    else:
        tgt_pos = matched.float()
    bce = F.binary_cross_entropy_with_logits(slot_out["existence"], tgt_pos, reduction="none")
    weights = torch.where(matched, torch.ones_like(bce), 0.1 * torch.ones_like(bce))
    terms["exist"] = (bce * weights).mean()

    # (b) type parameters (matched only): positivity BCE, level + geometry Huber (point estimates;
    # learned variance removed everywhere).
    pos_logit = expr["positive"][slot_idx]                   # [J, M]
    pos_tgt = tgt["positive"][type_idx].float()              # [J, M]
    mk = tgt["mk"]
    posbce = (F.binary_cross_entropy_with_logits(pos_logit, pos_tgt, reduction="none")
              * mk).sum() / (mk.sum() * J).clamp(min=1)
    lvl_tgt = logit(tgt["level"][type_idx])                  # normalised level -> logit space
    posmask = pos_tgt * mk
    lvl = (F.smooth_l1_loss(expr["level_mu"][slot_idx], lvl_tgt, reduction="none")
           * posmask).sum() / posmask.sum().clamp(min=1)
    gtgt = logit(tgt["geometry"][type_idx])                  # [J, 5]
    geom = F.smooth_l1_loss(slot_out["geometry"]["mu"][slot_idx], gtgt, reduction="none").mean()
    terms["type_param"] = posbce + lvl + geom

    # (c) context compositions: Dirichlet NLL per valid context (>= min_context_cells, J >= 2)
    ctx = A.new_zeros(())
    n_ctx = 0
    if J >= 2:
        alpha = slot_out["alpha"]                            # [K, C]
        for c in range(tgt["pi"].shape[0]):
            if tgt["context_cells"][c] >= min_context_cells:
                a_c = alpha[slot_idx, c]                     # [J], slots in type order
                ctx = ctx + dirichlet_nll(a_c, tgt["pi"][c, type_idx])
                n_ctx += 1
    terms["context"] = ctx / max(n_ctx, 1)

    # (d) token composition: soft CE between Abar (stride 16) and target in slot order
    Abar = slot_out["token_comp"].clamp(1e-6)               # [K+1, N16]
    Tt = token_comp_target(tgt["frac16"], slot_idx, K)      # [K+1, N16]
    tok_mask = tgt["token_mask"].float()                    # [N16]
    tokce = -(Tt * Abar.log()).sum(0)
    terms["token_comp"] = (tokce * tok_mask).sum() / tok_mask.sum().clamp(min=1)

    # (e) matched map: Dice + BCE at stride 4 (A is already a probability map)
    terms["matched_map"] = dice_bce_prob(A[slot_idx], Mgt[type_idx])

    match = dict(slot_idx=slot_idx, type_idx=type_idx)
    return terms, match


# --- discriminative loss over pure locations (f) -----------------------------------------------
def discriminative_one(emb, type4_pure, max_loc=4096, **kw):
    """Discriminative embedding loss for one sample: emb [N4, d]; type4_pure [N4] type index at
    pure stride-4 locations, -1 elsewhere. Subsamples to <= max_loc pure locations (plan 5.4)."""
    idx = (type4_pure >= 0).nonzero(as_tuple=True)[0]
    if idx.numel() < 2:
        return emb.new_zeros(())
    if idx.numel() > max_loc:
        idx = idx[torch.randperm(idx.numel(), device=emb.device)[:max_loc]]
    return discriminative_loss(emb[idx], type4_pure[idx], **kw)


# --- dense losses (g) --------------------------------------------------------------------------
def dense_losses(dense, tgt):
    """nucleus/cell BCE+Dice (ignore-masked), flow MSE inside visible cells, structure soft CE at
    stride 16. `dense` is the model's dense dict (batched); `tgt['dense']` holds the targets."""
    d = tgt["dense"]
    keep = (~d["ignore"]).float()[:, None]                   # [B,1,H,W]
    nuc = dice_bce(dense["nucleus"], d["nucleus"][:, None], mask=keep)
    cell = dice_bce(dense["cell"], d["cell"][:, None], mask=keep)
    vis = d["cell_visible"][:, None].float()                 # [B,1,H,W]
    flow = (((dense["flow"] - d["flow"]) ** 2) * vis).sum() / vis.sum().clamp(min=1) / 2
    logp = F.log_softmax(dense["structure"], dim=1)         # [B,C,Hg,Wg]
    smask = d["structure_mask"][:, None].float()
    struct = -((d["structure"] * logp).sum(1, keepdim=True) * smask).sum() / smask.sum().clamp(min=1)
    return dict(nucleus=nuc, cell=cell, flow=flow, structure=struct)


# --- global / per-marker losses (h) ------------------------------------------------------------
def global_marker_losses(out, tgt):
    # All point-estimate (Huber) losses now -- learned variance removed everywhere (a heteroscedastic
    # NLL let the heads under-fit the mean; uncertainty is deferred to a later calibration pass).
    g = tgt["global"]
    gerr = F.smooth_l1_loss(out["global_mu"], logit(g["u"]), reduction="none")
    if "mask" in g:                                          # padded globals (variable ARCH length)
        gm = g["mask"].float()
        gl = (gerr * gm).sum() / gm.sum().clamp(min=1)
    else:
        gl = gerr.mean()

    mk = tgt["marker"]
    present = mk["present"].bool()                           # [B,M]
    mo = out["marker"]
    nerr = F.smooth_l1_loss(mo["noise_mu"], logit(mk["noise_u"]), reduction="none")  # [B,M,P]
    pm = present[..., None].float()
    noise = (nerr * pm).sum() / pm.sum().clamp(min=1)
    # localisation mu: a Huber POINT-ESTIMATE loss in logit space, NOT an NLL. A learned variance
    # lets the head minimise a Gaussian NLL by predicting the mean with a huge variance (observed:
    # mu collapsed to the mean with uniformly large sigma and never learned), so here we optimise
    # accuracy directly. `mu_logvar` is still produced for later calibration but is not used here.
    mu_err = F.smooth_l1_loss(mo["mu"], logit(mk["mu_u"]), reduction="none")       # [B,M]
    mu = MU_LOSS_WEIGHT * (mu_err * present.float()).sum() / present.float().sum().clamp(min=1)
    terms = {"global": gl, "marker_noise": noise, "mu": mu}
    if "offset" in mo and "offset" in mk:
        off = ((mo["offset"] - mk["offset"]) ** 2)[present].mean()
        terms["offset"] = off
    return terms


# --- top-level combiner ------------------------------------------------------------------------
def total_loss(out, tgt, step, total_steps, w_match: MatchWeights = None,
               w: LossWeights = None, exist_target="one"):
    """Combine every group with the plan 5.3 fixed weights. Type terms (a-e) are matched and
    averaged over the T slot-attention iterations; (f) discriminative, (g) dense and (h)
    global/per-marker are charged once. `step` drives the seed-anchor weight (plan 5.4). Returns
    (scalar total, dict of logged term values)."""
    w = w or LossWeights()
    sched = schedule(step, total_steps)
    w_match = w_match or MatchWeights()
    w_match = MatchWeights(w_match.dice, w_match.bce, w_match.expr, w_match.exist, sched["w_seed"])

    B = out["global_mu"].shape[0]
    T = len(out["slots"])
    type_terms = {k: out["global_mu"].new_zeros(()) for k in
                  ("exist", "type_param", "context", "token_comp", "matched_map")}
    discrim = out["global_mu"].new_zeros(())
    for b in range(B):
        for ti in range(T):
            so = {k: (v[b] if torch.is_tensor(v) else
                      {kk: vv[b] for kk, vv in v.items()}) for k, v in out["slots"][ti].items()}
            terms, _ = type_losses_one(so, tgt["type"][b], w_match, w, exist_target)
            for k in type_terms:
                type_terms[k] = type_terms[k] + terms[k] / (B * T)
        discrim = discrim + discriminative_one(out["emb"][b], tgt["type"][b]["type4_pure"]) / B

    dense = dense_losses(out["dense"], tgt)
    gm = global_marker_losses(out, tgt)

    total = (w.exist * type_terms["exist"] + w.type_param * type_terms["type_param"]
             + w.context * type_terms["context"] + w.token_comp * type_terms["token_comp"]
             + w.matched_map * type_terms["matched_map"] + w.discrim * discrim
             + w.dense * sum(dense.values()) + w.global_marker * sum(gm.values()))

    log = {f"type/{k}": float(v) for k, v in type_terms.items()}
    log["type/discrim"] = float(discrim)
    log.update({f"dense/{k}": float(v) for k, v in dense.items()})
    log.update({f"gm/{k}": float(v) for k, v in gm.items()})
    log["sched/p_teacher"] = sched["p_teacher"]
    log["sched/w_seed"] = sched["w_seed"]
    return total, log


# --- geometric augmentation (plan 5.5) ---------------------------------------------------------
def augment_geometric(sample, k_rot, flip):
    """Label-consistent flip + 90-degree rotation of a whole sample dict (plan 5.5). Only geometry
    is touched -- NO intensity augmentation, which would change the expression targets. Rotates/
    flips the image and every label/fraction map the same way, and transforms the flow components,
    orientation angles (DIR_DEG and per-cell orientations) and per-channel offsets accordingly.

    `sample` keys handled if present: image [..,H,W], maps in `map_keys` [..,H,W], 'flow' [..,2,H,W]
    or [2,H,W], 'offsets' [...,2] (dy,dx), 'angles_deg' [...]. k_rot in {0,1,2,3}, flip bool (over W).
    """
    def _sp(x):                                              # flip/rotate the last two (H, W) axes
        if flip:
            x = torch.flip(x, dims=[-1])
        return torch.rot90(x, k_rot, dims=[-2, -1])

    out = dict(sample)
    for key in ("image",) + tuple(sample.get("map_keys", ())):
        if key in out:
            out[key] = _sp(out[key])

    if "flow" in out:
        fl = _sp(out["flow"])
        dy, dx = fl[..., 0, :, :], fl[..., 1, :, :]          # flow stored as (dy, dx)
        if flip:
            dx = -dx
        for _ in range(k_rot):                               # rot90 ccw: (dy,dx)->(-dx,dy)
            dy, dx = -dx, dy
        out["flow"] = torch.stack([dy, dx], dim=-3)

    if "offsets" in out:
        dy, dx = out["offsets"][..., 0].clone(), out["offsets"][..., 1].clone()
        if flip:
            dx = -dx
        for _ in range(k_rot):
            dy, dx = -dx, dy
        out["offsets"] = torch.stack([dy, dx], dim=-1)

    if "angles_deg" in out:
        a = out["angles_deg"].clone()
        if flip:
            a = (180.0 - a) % 360.0
        out["angles_deg"] = (a + 90.0 * k_rot) % 360.0
    return out
