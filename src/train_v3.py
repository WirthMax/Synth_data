"""v3 training step and evaluation diagnostics (plan Phase 6.2-6.3 core).

`train_step` runs one anchored-discovery step: teacher-seed a fraction `p_teacher` of the batch
(the rest fall back to the model's farthest-point sampling), forward, fill the seed anchors from the
seeds actually used, and take a clipped AdamW step. `evaluate_v3` reports the plan 6.2 diagnostics
with inference-style matching (w_seed = 0, no teacher seeds).

This is the training/eval CORE as plain functions; wiring it into the full round orchestration
(rounds/orchestrate.py, the proposal/policy loop, finalize) is the remaining Phase 6 integration and
is intentionally left to sit on top of these.
"""
import numpy as np
import torch

from losses_v3 import (MatchWeights, schedule, total_loss, match_one, dice_bce_prob,
                       _expr_pred)
from model_v3 import teacher_seeds
from targets_v3 import seed_frac_from


def move_targets(x, device):
    """Recursively move every tensor in the nested target structure (dicts, the per-sample `type`
    list) onto `device`, so the loss runs on the same device as the model outputs."""
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_targets(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_targets(v, device) for v in x]
    return x


def build_seeds(model, tgt, p_teacher, gen, device):
    """Per-sample seeds for a training step: teacher seeds (one per present type at a pure location)
    with probability `p_teacher`, otherwise `n_seeds = -1` so the model FPS-seeds that sample."""
    B = len(tgt["type"])
    seeds = torch.zeros(B, model.K, dtype=torch.long, device=device)
    n_seeds = torch.full((B,), -1, dtype=torch.long, device=device)
    for b in range(B):
        if float(torch.rand((), generator=gen)) >= p_teacher:
            continue
        t4 = tgt["type"][b]["type4_pure"].cpu()               # index selection stays on cpu
        pres = torch.unique(t4[t4 >= 0]).tolist()
        ts = teacher_seeds(t4, pres, gen)
        if ts:
            seeds[b, :len(ts)] = torch.as_tensor(ts[:model.K], device=device)
            n_seeds[b] = min(len(ts), model.K)
    return seeds, n_seeds


def train_step(model, opt, tile, present, tgt, step, total_steps, gen, device="cpu",
               family_weights=None):
    model.train()
    tgt = move_targets(tgt, device)
    sched = schedule(step, total_steps)
    seeds, n_seeds = build_seeds(model, tgt, sched["p_teacher"], gen, device)
    out = model(tile.to(device), present=present.to(device), strength=tgt["strength"],
                seeds=seeds, n_seeds=n_seeds)
    for b in range(len(tgt["type"])):
        tgt["type"][b]["seed_frac"] = seed_frac_from(
            out["seeds"][b], tgt["type"][b]["Mgt"], model.K)
    total, log = total_loss(out, tgt, step, total_steps, family_weights=family_weights)
    opt.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    log["loss/total"] = float(total)
    return log


@torch.no_grad()
def evaluate_v3(model, tile, present, tgt, device="cpu"):
    """Plan 6.2 diagnostics on one (already collated) batch, with inference-style matching
    (w_seed = 0, FPS seeds). Returns a dict of scalar means over the batch."""
    model.eval()
    tgt = move_targets(tgt, device)
    out = model(tile.to(device), present=present.to(device), strength=tgt["strength"])  # FPS seeds
    w0 = MatchWeights(seed=0.0)
    B = len(tgt["type"])
    acc = dict(recovered_rate=[], type_count_err=[], mu_nmse=[], compartment_acc=[],
               noise_nmse=[], global_nmse=[], positivity_acc=[], expr_nmse=[], geom_nmse=[],
               cell_iou=[], mean_match_cost=[], duplicate_rate=[])
    slots = out["slots"][-1]
    for b in range(B):
        t = tgt["type"][b]
        Mgt, J = t["Mgt"], t["Mgt"].shape[0]
        M = t["mk"].shape[0]
        maps = slots["maps"][b]
        A = maps[:model.K]
        expr = {k: v[b][:, :M] for k, v in slots["expression"].items()}
        p_exist = torch.sigmoid(slots["existence"][b])
        E_pred = _expr_pred(expr)

        # type-count error is defined even for an empty tile (the model should predict zero types);
        # duplicate rate too. The matching-based metrics (recovered/positivity/expr/geom/cost) have
        # no referent when J=0, so they are skipped for that sample.
        acc["type_count_err"].append(abs(int((p_exist > 0.5).sum()) - J))
        mass = A.sum(1) / A.sum(1).sum().clamp(min=1e-6)
        if J > 0:
            slot_idx, type_idx = match_one(A, Mgt, E_pred, t["E_gt"], t["mk"], p_exist,
                                           torch.zeros(model.K, J, device=device), w0)
            # recovered-type rate: matched slot Dice >= 0.5
            dice = (2 * (A[slot_idx] * Mgt[type_idx]).sum(1)) / (
                A[slot_idx].sum(1) + Mgt[type_idx].sum(1)).clamp(min=1e-6)
            acc["recovered_rate"].append(float((dice >= 0.5).float().mean()))
            matched = torch.zeros(model.K, dtype=torch.bool, device=device)
            matched[slot_idx] = True
            pos = (torch.sigmoid(expr["positive"][slot_idx]) > 0.5).float()
            acc["positivity_acc"].append(float((pos == t["positive"]).float().mean()))
            acc["expr_nmse"].append(float(((E_pred[slot_idx] - t["E_gt"]) ** 2).mean()))
            gmu = slots["geometry"]["mu"][b][slot_idx]
            acc["geom_nmse"].append(float(((torch.sigmoid(gmu) - t["geometry"]) ** 2).mean()))
            acc["mean_match_cost"].append(float(1 - dice.mean()))
        else:
            matched = torch.zeros(model.K, dtype=torch.bool, device=device)
        # duplicate rate: confident slots carrying > 1% of mass that are not matched to any type
        dup = (~matched) & (p_exist > 0.5) & (mass > 0.01)
        acc["duplicate_rate"].append(float(dup.float().mean()))

        cl = (torch.sigmoid(out["dense"]["cell"][b, 0]) > 0.5)
        ct = t if False else tgt["dense"]["cell"][b] > 0.5
        inter = (cl & ct).sum().float()
        acc["cell_iou"].append(float(inter / (cl | ct).sum().clamp(min=1)))

    # localisation mu: regression quality, plus a derived-compartment accuracy for continuity with
    # the old metric (threshold predicted and true mu into nuclear/cyto/membrane). Constants from
    # config (not flexible_types) so this stays importable in the synmodel env.
    import config as cfg
    cm = tgt["marker"]
    pr = cm["present"].bool()
    mu_pred_u = torch.sigmoid(out["marker"]["mu"])              # normalised [0,1]
    mu_tgt_u = cm["mu_u"]
    acc["mu_nmse"] = [float(((mu_pred_u[pr] - mu_tgt_u[pr]) ** 2).mean())]
    span = cfg.MU_HI - cfg.MU_LO
    mp = (mu_pred_u[pr] * span + cfg.MU_LO).cpu().numpy()
    mt = (mu_tgt_u[pr] * span + cfg.MU_LO).cpu().numpy()
    centres = np.array(list(cfg.COMPARTMENT_MU.values()))
    bp = np.abs(mp[:, None] - centres[None]).argmin(1)
    bt = np.abs(mt[:, None] - centres[None]).argmin(1)
    acc["compartment_acc"] = [float((bp == bt).mean())]

    # per-family regression quality for the other point-estimate heads (normalised [0,1] space)
    noise_pred = torch.sigmoid(out["marker"]["noise_mu"])[pr]   # [n_present, P]
    acc["noise_nmse"] = [float(((noise_pred - cm["noise_u"][pr]) ** 2).mean())]
    g = tgt["global"]
    gmask = g["mask"].bool() if "mask" in g else torch.ones_like(g["u"], dtype=torch.bool)
    gpred = torch.sigmoid(out["global_mu"])
    acc["global_nmse"] = [float(((gpred - g["u"]) ** 2)[gmask].mean())]
    return {k: float(np.mean(v)) for k, v in acc.items()}
