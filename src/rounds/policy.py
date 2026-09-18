"""Section 7: the targeting policy. `decide()` is pure and deterministic -- no randomness, no
I/O -- so it's testable with plain dicts and safe to call speculatively (--dry-run). All state
(saturation) is threaded explicitly: it comes IN via `targeting_history["saturation_state"]` and
goes OUT via `decision_log["saturation_state"]`; the caller (orchestrate.py) is what persists it
between rounds, which is also how a manual reset or a phase change works -- just pass in
different state, decide() has no memory of its own.

stdlib + numpy only, no scipy/torch -- policy.py needs to be importable from anywhere.
"""
import _pathfix  # noqa: F401

import argparse
import glob
import json
import os
import re

import numpy as np
import yaml

from proposal import Proposal


def _default_policy_cfg(config):
    p = dict(top_m=3, nmse_floor=0.05, delta_min=0.02, patience=2, max_ratio=3.0,
            min_bin_count=30, prior_mix=0.3, ema=0.5, exclude=[])
    p.update(config.get("policy", {}))
    return p


def _default_proposal_cfg(config):
    p = dict(alpha_prior=0.5, bins=10, widen=False)
    p.update(config.get("proposal", {}))
    return p


def _update_saturation_state(eval_history, targeting_history, policy_cfg):
    """Advances the carried-in saturation state by exactly ONE round -- the transition from the
    round before the latest eval to the latest eval -- never recomputed from the whole history,
    so a caller's manual reset (or a fresh state at a phase change) genuinely takes effect
    instead of being immediately overridden by old history on the next call.
    """
    state = {k: dict(v) for k, v in targeting_history.get("saturation_state", {}).items()}
    picks_by_round = targeting_history.get("picks_by_round", [])
    lag_by_round = targeting_history.get("lag_by_round", [])
    r_last = len(eval_history) - 1
    log_entries = []
    if r_last < 1 or r_last >= len(picks_by_round):
        return state, log_entries
    lag = lag_by_round[r_last] if r_last < len(lag_by_round) else 0
    before_idx = r_last - 1 - lag
    if before_idx < 0:
        return state, log_entries
    before_eval, after_eval = eval_history[before_idx], eval_history[r_last]
    for comp in picks_by_round[r_last]:
        k = comp["param"]
        if k not in before_eval or k not in after_eval:
            continue
        delta = before_eval[k]["nmse"] - after_eval[k]["nmse"]
        ci_lo, ci_hi = after_eval[k]["nmse_ci"]
        ci_half_width = (ci_hi - ci_lo) / 2.0
        threshold = max(policy_cfg["delta_min"], ci_half_width)
        st = state.setdefault(k, {"saturated": False, "streak": 0})
        if not st["saturated"]:
            if delta < threshold:
                st["streak"] += 1
            else:
                st["streak"] = 0
            newly_saturated = st["streak"] >= policy_cfg["patience"]
            if newly_saturated:
                st["saturated"] = True
            log_entries.append(dict(param=k, delta=delta, threshold=threshold, lag=lag,
                                    streak=st["streak"], newly_saturated=newly_saturated))
    return state, log_entries


def _previous_bin_weights(path, targeting_history):
    picks_by_round = targeting_history.get("picks_by_round", [])
    if not picks_by_round:
        return None
    for comp in picks_by_round[-1]:
        if comp["param"] == path:
            return comp["bin_weights"]
    return None


def _bin_weights_for_pick(path, latest_eval, targeting_history, policy_cfg):
    stats = latest_eval[path]
    binned_mse, binned_count = stats["binned_mse"], stats["binned_count"]
    bins = len(binned_mse)
    prior_mass = 1.0 / bins
    mse_k = max(stats["mse"], 1e-12)
    max_ratio = policy_cfg["max_ratio"]

    g_raw = np.empty(bins)
    for b in range(bins):
        if binned_count[b] < policy_cfg["min_bin_count"] or binned_mse[b] is None:
            ratio = 1.0
        else:
            ratio = float(np.clip(binned_mse[b] / mse_k, 1.0 / max_ratio, max_ratio))
        g_raw[b] = prior_mass * ratio
    g_raw = g_raw / g_raw.sum()

    p_binned = np.full(bins, prior_mass)
    g = (1 - policy_cfg["prior_mix"]) * g_raw + policy_cfg["prior_mix"] * p_binned

    prev_bw = _previous_bin_weights(path, targeting_history)
    if prev_bw is not None:
        g = policy_cfg["ema"] * np.asarray(prev_bw) + (1 - policy_cfg["ema"]) * g

    g = g / g.sum()

    upweighted = np.where(g > prior_mass * 1.01)[0]
    edge_warning = len(upweighted) > 0 and set(upweighted.tolist()) <= {0, bins - 1}
    return g.tolist(), bool(edge_warning)


def decide(eval_history, targeting_history, config):
    """eval_history: list of per-round `per_parameter` dicts (evaluate.py's report section),
    one entry per round evaluated so far, in round order.
    targeting_history: {"picks_by_round": [[{"param":..,"bin_weights":[...]}], ...],
                        "lag_by_round": [int, ...], "saturation_state": {...}}
    config: the run's full config dict (policy.*, proposal.* keys read from it).
    Returns (proposal_dict, decision_log). `proposal_dict` is a Proposal.to_dict() -- the caller
    constructs `Proposal.from_dict(proposal_dict)` and saves it.
    """
    if not eval_history:
        raise ValueError("decide() needs at least one round of eval_history")
    policy_cfg = _default_policy_cfg(config)
    proposal_cfg = _default_proposal_cfg(config)
    next_round = len(eval_history)

    saturation_state, saturation_log = _update_saturation_state(eval_history, targeting_history,
                                                                 policy_cfg)
    latest = eval_history[-1]
    exclude = set(policy_cfg["exclude"])
    nmse_floor = policy_cfg["nmse_floor"]

    per_parameter_log = {}
    candidates = []
    for k, stats in latest.items():
        if k in exclude:
            status = "excluded"
        elif saturation_state.get(k, {}).get("saturated", False):
            status = "saturated"
        elif stats["nmse"] <= nmse_floor:
            status = "below_floor"
        else:
            status = "candidate"
            candidates.append(k)
        per_parameter_log[k] = dict(nmse=stats["nmse"], nmse_ci=stats["nmse_ci"], status=status)

    warnings = []
    if not candidates:
        proposal = Proposal.prior_only(next_round, alpha_prior=proposal_cfg["alpha_prior"],
                                       bins=proposal_cfg["bins"], widen=proposal_cfg["widen"])
        picks_for_history = []
    else:
        top_m = policy_cfg["top_m"]
        picks_paths = sorted(candidates, key=lambda k: (-latest[k]["nmse"], k))[:top_m]
        picks = []
        picks_for_history = []
        for k in picks_paths:
            bw, edge_warning = _bin_weights_for_pick(k, latest, targeting_history, policy_cfg)
            picks.append((k, bw))
            picks_for_history.append(dict(param=k, bin_weights=bw))
            per_parameter_log[k]["status"] = "picked"
            per_parameter_log[k]["bin_weights"] = bw
            per_parameter_log[k]["edge_bin_warning"] = edge_warning
            if edge_warning:
                warnings.append(
                    f"{k}: every upweighted bin is an edge bin -- for a weakly identified "
                    f"parameter, larger error at the range edges is partly expected shrinkage "
                    f"toward the prior mean, not necessarily something more data will fix"
                )
        proposal = Proposal.targeted(next_round, picks, alpha_prior=proposal_cfg["alpha_prior"],
                                     bins=proposal_cfg["bins"], widen=proposal_cfg["widen"])

    decision_log = dict(
        round_decided_for=next_round, n_candidates=len(candidates),
        picks=[c["param"] for c in picks_for_history], saturation_updates=saturation_log,
        saturation_state=saturation_state, per_parameter=per_parameter_log, warnings=warnings,
    )
    return proposal.to_dict(), decision_log


def format_decision_log(decision_log):
    lines = [f"round {decision_log['round_decided_for']}: "
            f"{decision_log['n_candidates']} candidate(s), picks={decision_log['picks']}"]
    for w in decision_log["warnings"]:
        lines.append(f"  WARNING: {w}")
    for su in decision_log["saturation_updates"]:
        tag = " -> SATURATED" if su["newly_saturated"] else ""
        lines.append(f"  {su['param']}: delta={su['delta']:.4f} threshold={su['threshold']:.4f} "
                    f"streak={su['streak']}{tag}")
    for path, d in sorted(decision_log["per_parameter"].items(),
                          key=lambda kv: -kv[1]["nmse"]):
        extra = f" bins={d['bin_weights']}" if d["status"] == "picked" else ""
        lines.append(f"  {d['status']:14s} nmse={d['nmse']:7.4f} "
                    f"ci=[{d['nmse_ci'][0]:.4f},{d['nmse_ci'][1]:.4f}]  {path}{extra}")
    return "\n".join(lines)


def _load_eval_history(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "eval", "round_*.json")))
    history = []
    for fp in files:
        with open(fp) as f:
            report = json.load(f)
        history.append(report["per_parameter"])
    return history


def _load_targeting_history(run_dir, n_rounds):
    """`proposals/round_{r:03d}.json` is the proposal that PRODUCED round r -- round 0 always has
    none (it's prior-only by construction, decided before any eval exists), so picks_by_round
    must be indexed by the round number embedded in each filename, not by naive glob-list
    position, or every entry silently shifts by one round once round 0's file is (correctly)
    missing.
    """
    picks_by_round = [[] for _ in range(n_rounds)]
    lag_by_round = [0] * n_rounds
    pattern = re.compile(r"round_(\d{3})\.json$")
    for fp in sorted(glob.glob(os.path.join(run_dir, "proposals", "round_*.json"))):
        m = pattern.search(fp)
        if not m:
            continue
        r = int(m.group(1))
        if r >= n_rounds:
            continue  # a proposal decided for a round that never actually ran (e.g. stopped early)
        with open(fp) as f:
            d = json.load(f)
        picks_by_round[r] = [dict(param=c["param"], bin_weights=c["bin_weights"])
                             for c in d["components"] if c["kind"] == "reshape"]
    state_path = os.path.join(run_dir, "policy_state.json")
    saturation_state = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            saturation_state = json.load(f)
    return dict(picks_by_round=picks_by_round, lag_by_round=lag_by_round,
               saturation_state=saturation_state)


def run_policy(run_dir, config, dry_run=False):
    eval_history = _load_eval_history(run_dir)
    targeting_history = _load_targeting_history(run_dir, len(eval_history))
    proposal_dict, decision_log = decide(eval_history, targeting_history, config)
    print(format_decision_log(decision_log))
    if not dry_run:
        os.makedirs(os.path.join(run_dir, "proposals"), exist_ok=True)
        Proposal.from_dict(proposal_dict).save(
            os.path.join(run_dir, "proposals", f"round_{proposal_dict['round']:03d}.json"))
        with open(os.path.join(run_dir, "policy_state.json"), "w") as f:
            json.dump(decision_log["saturation_state"], f, indent=1)
        decisions_dir = os.path.join(run_dir, "decisions")
        os.makedirs(decisions_dir, exist_ok=True)
        with open(os.path.join(decisions_dir,
                               f"round_{proposal_dict['round']:03d}.json"), "w") as f:
            json.dump(decision_log, f, indent=1)
    return proposal_dict, decision_log


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    with open(a.config) as f:
        config = yaml.safe_load(f)
    run_policy(a.run_dir, config, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
