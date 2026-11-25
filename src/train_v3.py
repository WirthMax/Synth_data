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
from model_v3 import teacher_seeds, decode_queries
from targets_v3 import seed_frac_from
from flow_decode import flow_to_instances, match


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
               family_weights=None, accum=(0, 1)):
    """One forward/backward. `accum` = (i, n): micro-batch i of n accumulated into one optimiser
    step -- the gradient is zeroed before i = 0, the loss scaled by 1/n, and the step (with the
    clip) taken after i = n-1. A 512 px region fits the GPU one at a time, so a batch of regions is
    accumulated; (0, 1) is the plain step."""
    i_acc, n_acc = accum
    model.train()
    tgt = move_targets(tgt, device)
    sched = schedule(step, total_steps)
    seeds, n_seeds = build_seeds(model, tgt, sched["p_teacher"], gen, device)
    out = model(tile.to(device), present=present.to(device), strength=tgt["strength"],
                seeds=seeds, n_seeds=n_seeds, ref=tgt.get("ref"))
    for b in range(len(tgt["type"])):
        tgt["type"][b]["seed_frac"] = seed_frac_from(
            out["seeds"][b], tgt["type"][b]["Mgt"], model.K)
    total, log = total_loss(out, tgt, step, total_steps, family_weights=family_weights)
    if i_acc == 0:
        opt.zero_grad()
    (total / n_acc).backward()
    if i_acc == n_acc - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    log["loss/total"] = float(total)
    return log


# visible-cell bins for the instance metrics: a per-tile mean is dominated by the many sparse tiles,
# and hid that the query head falls from F1 0.74 (1-10 cells) to 0.18 (> 100 cells) -- the regime
# confluent real tissue sits in. Bin edges are inclusive upper bounds; 0 = empty tiles.
DENSITY_BINS = ((0, 0), (1, 10), (11, 50), (51, 100), (101, 10 ** 9))


def density_bin(n):
    return next(i for i, (lo, hi) in enumerate(DENSITY_BINS) if lo <= n <= hi)


def eval_tags(model, B, M, seed, device):
    """Seeded marker tags for evaluation. The model draws fresh random tags on every forward, so the
    same weights scored +-0.03-0.05 cell F1 apart between rounds; fixing them per batch makes the
    eval a function of the weights alone."""
    import math
    g = torch.Generator().manual_seed(int(seed))
    return (torch.randn(B, M, model.d_tag, generator=g) / math.sqrt(model.d_tag)).to(device)


@torch.no_grad()
def evaluate_v3(model, tile, present, tgt, device="cpu", tag_seed=None, decode=None):
    """Plan 6.2 diagnostics on one (already collated) batch, with inference-style matching
    (w_seed = 0, FPS seeds). Returns a dict of scalar means over the batch, plus `_per_tile`: one
    record per sample (visible cells, decoded cells, query-head cell / nucleus F1) that the caller
    pools over the WHOLE eval set into density bins and a signed count error. `tag_seed` fixes the
    marker tags (None = fresh random tags, the old behaviour)."""
    model.eval()
    tgt = move_targets(tgt, device)
    B0, M0 = tile.shape[:2]
    tags = None if tag_seed is None else eval_tags(model, B0, M0, tag_seed, device)
    out = model(tile.to(device), present=present.to(device), strength=tgt["strength"],
                tags=tags, ref=tgt.get("ref"))                                      # FPS seeds
    w0 = MatchWeights(seed=0.0)
    B = len(tgt["type"])
    acc = dict(recovered_rate=[], type_count_err=[], mu_nmse=[], compartment_acc=[],
               noise_nmse=[], global_nmse=[], positivity_acc=[], expr_nmse=[], geom_nmse=[],
               cell_iou=[], mean_match_cost=[], duplicate_rate=[])
    per_tile = []
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

        # instance-level: decode the predicted flows exactly as inference does and score against the
        # stored labels. cell_iou alone hid this -- 0.77 semantic IoU sat on a 0.31 instance F1.
        # Nuclei are scored on visible cells' nuclei (the stored labels are visible-only).
        if "inst" in tgt["dense"]:
            gi = tgt["dense"]["inst"][b].cpu().numpy()
            gn = np.where(tgt["dense"]["nucleus"][b].cpu().numpy() > 0.5, gi, 0)
            dn = out["dense"]
            if "flow" in dn:     # flow baseline only while the model still trains flows
                pc = flow_to_instances(dn["flow"][b], torch.sigmoid(dn["cell"][b, 0]) > 0.5, device=device)
                pn = flow_to_instances(dn["nuc_flow"][b], torch.sigmoid(dn["nucleus"][b, 0]) > 0.5,
                                       device=device)
                acc.setdefault("cell_f1", []).append(match(gi, pc)["f1"])
                acc.setdefault("nuc_f1", []).append(match(gn, pn)["f1"])

    # the query head, scored per WINDOW (one per sample on a tile; the covering grid of a region,
    # each against the truth cut to it, so a per-window record means what a per-tile one did and
    # the density bins stay "cells per 52 um"). Slivers the window edge cuts off a cell are taken
    # out of the truth, as in the loss (losses_v3.window_targets).
    if out.get("inst") is not None and "inst" in tgt["dense"]:
        from losses_v3 import window_targets
        S = out["inst"]["cell"].shape[-1]
        for i, (b, y0, x0) in enumerate(out["inst"]["win"].tolist()):
            g, nu, _, _ = window_targets(tgt["dense"]["inst"][b], tgt["dense"]["nucleus"][b] > 0.5,
                                         tgt["dense"]["ignore"][b], {}, y0, x0, S)
            gi = g.cpu().numpy()
            gn = np.where(nu.cpu().numpy(), gi, 0)
            qc, qn, props = decode_queries(out["inst"], i, **(decode or {}))
            mc, mn = match(gi, qc), match(gn, qn)
            acc.setdefault("q_cell_f1", []).append(mc["f1"])
            acc.setdefault("q_nuc_f1", []).append(mn["f1"])
            acc.setdefault("q_count_err", []).append(abs(mc["n_pred"] - mc["n_gt"]))
            per_tile.append(dict(n_gt=mc["n_gt"], n_pred=mc["n_pred"], cell_f1=mc["f1"],
                                 nuc_f1=mn["f1"], cell_prec=mc["prec"], cell_rec=mc["rec"]))

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
    # a metric with no referent in this batch (the matching metrics of an EMPTY tile) is left out,
    # not reported as nan: at batch 1 (a 512 px region) an empty region would otherwise turn the
    # pooled mean, and with it the orchestrator's composite score, into nan for good
    res = {k: float(np.mean(v)) for k, v in acc.items() if len(v)}
    res["_per_tile"] = per_tile
    return res


def pool_instance_metrics(per_tile):
    """Whole-eval-set instance metrics from the `_per_tile` records of every batch:
    - q_count_lr: mean SIGNED log((pred + 1) / (true + 1)); > 0 = over-segmentation. The old
      `q_count_err` was unsigned, so over- and under-calling looked the same.
    - q_count_alr: mean |log ratio|, the scale-free error the composite can use;
    - q_cell_f1_b{i} / q_nuc_f1_b{i}: F1 per visible-cell bin (DENSITY_BINS), with the bin's tile
      count n_b{i} and its signed count error q_count_lr_b{i};
    - q_cell_f1_macro / q_nuc_f1_macro: the mean over the NON-EMPTY bins, so dense tissue counts as
      much as sparse tissue however rare it is in the prior."""
    if not per_tile:
        return {}
    out = {}
    lr = np.array([np.log((r["n_pred"] + 1) / (r["n_gt"] + 1)) for r in per_tile])
    out["q_count_lr"], out["q_count_alr"] = float(lr.mean()), float(np.abs(lr).mean())
    bins = np.array([density_bin(r["n_gt"]) for r in per_tile])
    cf = np.array([r["cell_f1"] for r in per_tile])
    nf = np.array([r["nuc_f1"] for r in per_tile])
    mc, mn = [], []
    for i in range(len(DENSITY_BINS)):
        sel = bins == i
        out[f"n_b{i}"] = int(sel.sum())
        if sel.any():
            out[f"q_cell_f1_b{i}"] = float(cf[sel].mean())
            out[f"q_nuc_f1_b{i}"] = float(nf[sel].mean())
            out[f"q_count_lr_b{i}"] = float(lr[sel].mean())
            if i > 0:                       # the macro average is over tiles WITH cells
                mc.append(out[f"q_cell_f1_b{i}"])
                mn.append(out[f"q_nuc_f1_b{i}"])
    if mc:
        out["q_cell_f1_macro"], out["q_nuc_f1_macro"] = float(np.mean(mc)), float(np.mean(mn))
    return out
