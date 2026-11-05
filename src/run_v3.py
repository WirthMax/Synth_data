"""v3 training driver: generate flexible-mode bundle shards, and train ModelV3 on them.

Two subcommands, run in the two envs (the split is the scipy boundary, as everywhere in v3):

    # generate data (synmac -- needs the generation stack + scipy)
    /mnt/Vol_b/miniconda3/envs/synmac/bin/python run_v3.py gen --n 2000 --out ../data/v3 --workers 8

    # train (synmodel -- torch)
    /mnt/Vol_b/miniconda3/envs/synmodel/bin/python run_v3.py train --data ../data/v3 --out runs/v3a \
        --steps 20000 --batch-size 4 --device cuda

`gen` writes one `bundle_XXXXXX.npz` per sample (flexible_bundle.build_bundle). The curriculum
(plan 3.7) is set by `--progress` in [0,1] (0 = gentle start prior, 1 = full final prior); a real
schedule regenerates at a rising progress between rounds. Heavy imports are function-local so each
subcommand only needs its own env.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_GLOBAL = 80          # pad length: 4 prepended DENSITY scalars + variable ARCH (max ~65 at 3
                       # structures) + TG.SIZE_SIGMA. Was 72 before the density targets were added.
K_SLOTS = 30           # type slots (plan default)


# ---------------- generate ----------------
def _gen_one(args):
    i, seed, progress, outdir, with_artifacts = args
    import config as cfg
    import flexible_bundle as fb
    cur = cfg.curriculum_at(progress)
    b = fb.build_bundle(i, seed=seed + i, curriculum=cur, with_artifacts=with_artifacts)
    np.savez(os.path.join(outdir, f"bundle_{i:06d}.npz"), **b)
    return i, int(b["th_M"]), int(b["th_K"]), int(b["present"].sum())


def cmd_gen(a):
    os.makedirs(a.out, exist_ok=True)
    jobs = [(i, a.seed, a.progress, a.out, a.with_artifacts) for i in range(a.n)]
    t0 = time.time()
    done = 0
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, M, K, J in ex.map(_gen_one, jobs, chunksize=4):
                done += 1
                if done % 50 == 0 or done == a.n:
                    print(f"  {done}/{a.n}  ({(time.time()-t0)/done:.2f}s/sample)", flush=True)
    else:
        for job in jobs:
            i, M, K, J = _gen_one(job)
            done += 1
            if done % 50 == 0 or done == a.n:
                print(f"  {done}/{a.n}", flush=True)
    print(f"wrote {a.n} bundles to {a.out} in {time.time()-t0:.0f}s")


# ---------------- train ----------------
def _cosine_lr(step, total, base_lr, warmup):
    import math
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(p, 1.0)))


def cmd_train(a):
    import torch
    from torch.utils.data import Subset, DataLoader

    import targets_v3 as T
    import train_v3 as TR
    from data_v3 import BundleShardDataset
    from model_v3 import ModelV3

    os.makedirs(a.out, exist_ok=True)
    device = a.device
    ds = BundleShardDataset(a.data, k_slots=a.k_slots, n_global=a.n_global)
    n = len(ds)
    n_val = max(1, int(a.val_frac * n))
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    coll = lambda s: T.collate_v3(s, m_max=a.m_max)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=a.batch_size, shuffle=True,
                              num_workers=a.num_workers, collate_fn=coll, drop_last=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=a.batch_size, shuffle=False,
                            num_workers=a.num_workers, collate_fn=coll)
    print(f"dataset: {n} samples ({len(train_idx)} train / {len(val_idx)} val)", flush=True)

    model = ModelV3(n_global=a.n_global, n_marker_noise=4, K=a.k_slots, T=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    fw = json.loads(a.family_weights) if a.family_weights else None

    start_step = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=device)
        # SHAPE-TOLERANT warm-start: a checkpoint from before a head grew (e.g. the global head when
        # the density targets were added, 72 -> 80) has mismatched tensors. Load only the ones whose
        # shape still matches; the rest keep their fresh init. load_state_dict(strict=False) ignores
        # missing/unexpected keys but still errors on a SIZE mismatch, so filter first.
        cur = model.state_dict()
        compatible = {k: v for k, v in ck["model"].items()
                      if k in cur and cur[k].shape == v.shape}
        skipped = [k for k in ck["model"] if k not in compatible]
        model.load_state_dict(compatible, strict=False)
        if skipped:
            print(f"warm-start: loaded {len(compatible)}/{len(ck['model'])} tensors; "
                  f"re-initialised {skipped} (shape changed) and started a fresh optimizer", flush=True)
        elif "opt" in ck:
            opt.load_state_dict(ck["opt"])     # shapes all match -> continue the optimizer state too
        start_step = int(ck["step"])
        print(f"resumed {a.resume} at step {start_step}", flush=True)
    total_steps = a.total_steps or a.steps        # grand total (schedule denominator across rounds)

    gen = torch.Generator().manual_seed(a.seed + start_step)
    logf = open(os.path.join(a.out, "train_log.jsonl"), "a")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ModelV3: {n_params/1e6:.1f}M params | device={device} | "
          f"fw={fw} | cum step {start_step}->{start_step+a.steps}/{total_steps}", flush=True)

    local, t0 = 0, time.time()
    while local < a.steps:
        for tile, present, tgt in train_loader:
            cum = start_step + local
            lr = _cosine_lr(cum, total_steps, a.lr, a.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            log = TR.train_step(model, opt, tile, present, tgt, cum, total_steps, gen,
                                device=device, family_weights=fw)
            if local % a.log_every == 0:
                log["step"], log["lr"] = cum, lr
                print(f"step {cum:6d}  loss {log['loss/total']:9.2f}  lr {lr:.2e}  "
                      f"({(time.time()-t0)/(local+1):.2f}s/step)", flush=True)
                logf.write(json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                       for k, v in log.items()}) + "\n")
                logf.flush()
            if local > 0 and local % a.eval_every == 0:
                _run_eval(model, val_loader, device, cum, logf)
            if local > 0 and local % a.ckpt_every == 0:
                _save(model, opt, a, cum)
            local += 1
            if local >= a.steps:
                break
    end_step = start_step + a.steps
    _save(model, opt, a, end_step)
    agg = _run_eval(model, val_loader, device, end_step, logf)
    json.dump(agg, open(os.path.join(a.out, "metrics.json"), "w"))     # for the orchestrator
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


def _run_eval(model, val_loader, device, step, logf):
    import numpy as np
    import train_v3 as TR
    mets = []
    for tile, present, tgt in val_loader:
        mets.append(TR.evaluate_v3(model, tile, present, tgt, device=device))
    agg = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]} if mets else {}
    print(f"  [eval @ {step}] " + "  ".join(f"{k}={v:.3f}" for k, v in agg.items()), flush=True)
    logf.write(json.dumps({"eval_step": step, **{f"eval/{k}": round(v, 5)
                                                 for k, v in agg.items()}}) + "\n")
    logf.flush()
    return agg


def _save(model, opt, a, step):
    import torch
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step,
                    cfg=dict(n_global=a.n_global, k_slots=a.k_slots)),
               os.path.join(a.out, "model_last.pt"))


# ---------------- rounds (Phase 6 orchestrator) ----------------
# which validation metric reflects each adaptively-weighted parameter family (plan 6.3). Only the
# global + per-marker scalar families get reshaped; type-set difficulty is driven by the curriculum,
# not the loss weights -- but level/geom (per-type regressions) are cheap to include and harmless.
_FAMILY_METRIC = {"mu": "mu_nmse", "noise": "noise_nmse", "global": "global_nmse",
                  "level": "expr_nmse", "geom": "geom_nmse"}


def _bundle_count(d):
    import glob
    return len(glob.glob(os.path.join(d, "bundle_*.npz")))


def update_family_weights(metrics, hard_focus):
    """Upweight the families the model fits worst (normalised so the mean weight is ~1), clipped to a
    sane band. `hard_focus` scales the strength; 0 = uniform. Pure function (unit-tested)."""
    errs = {f: metrics[m] for f, m in _FAMILY_METRIC.items() if metrics.get(m) is not None}
    if not errs or hard_focus <= 0:
        return {}
    mean = sum(errs.values()) / len(errs)
    return {f: float(min(3.0, max(0.5, (e / (mean + 1e-9)) ** hard_focus))) for f, e in errs.items()}


def cmd_rounds(a):
    import subprocess, shutil
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.data_root, exist_ok=True)
    grand_total = a.rounds * a.spr_steps + a.final_steps
    ckpt = os.path.join(a.out, "model_last.pt")
    rounds_log = os.path.join(a.out, "rounds_log.jsonl")
    t0 = time.time()

    def past_deadline():
        return a.max_hours and (time.time() - t0) / 3600.0 >= a.max_hours

    def gen(data_dir, progress, seed):
        have = _bundle_count(data_dir)
        if have >= a.spr:
            print(f"[rounds] {data_dir} has {have} bundles, skipping gen", flush=True)
            return
        subprocess.run([a.synmac_python, os.path.abspath(__file__), "gen", "--n", str(a.spr),
                        "--out", data_dir, "--progress", f"{progress}",
                        "--workers", str(a.gen_workers), "--seed", str(seed)], check=True)

    def train(data_dir, steps, fw, seed):
        cmd = [a.synmodel_python, os.path.abspath(__file__), "train", "--data", data_dir,
               "--out", a.out, "--steps", str(steps), "--total-steps", str(grand_total),
               "--batch-size", str(a.batch_size), "--device", a.device,
               "--num-workers", str(a.num_workers), "--seed", str(seed),
               "--eval-every", "1000", "--ckpt-every", "2000"]
        if os.path.exists(ckpt):
            cmd += ["--resume", ckpt]
        if fw:
            cmd += ["--family-weights", json.dumps(fw)]
        subprocess.run(cmd, check=True)

    # resume-aware: a state file tracks completed rounds / adaptive weights / final flag, so a
    # relaunch (after a deadline stop or a crash) continues the curriculum where it left off rather
    # than replaying from round 0. The model itself resumes from model_last.pt.
    state_path = os.path.join(a.out, "rounds_state.json")
    state = (json.load(open(state_path)) if os.path.exists(state_path)
             else {"completed": 0, "final_done": False, "fw": {}})
    fw = state["fw"]
    if state["completed"]:
        print(f"[rounds] resuming orchestration: {state['completed']}/{a.rounds} rounds done, "
              f"final_done={state['final_done']}", flush=True)

    def _save_state():
        json.dump(state, open(state_path, "w"))

    for r in range(state["completed"], a.rounds):
        if past_deadline():
            print(f"[rounds] deadline ({a.max_hours}h) reached, stopping before round {r}", flush=True)
            break
        progress = r / a.rounds                         # rising curriculum; final fine-tune hits 1.0
        data_dir = os.path.join(a.data_root, f"round_{r}")
        print(f"\n[rounds] ===== ROUND {r}/{a.rounds}  progress={progress:.2f}  "
              f"elapsed={ (time.time()-t0)/3600:.1f}h  fw={fw} =====", flush=True)
        gen(data_dir, progress, a.seed + r * 10_000)
        train(data_dir, a.spr_steps, fw, a.seed + r)
        metrics = json.load(open(os.path.join(a.out, "metrics.json")))
        fw = update_family_weights(metrics, a.hard_focus)
        with open(rounds_log, "a") as f:
            f.write(json.dumps({"round": r, "progress": progress,
                                "elapsed_h": round((time.time() - t0) / 3600, 2),
                                "metrics": {k: round(v, 5) for k, v in metrics.items()},
                                "next_family_weights": fw}) + "\n")
        if not a.keep_data:
            shutil.rmtree(data_dir, ignore_errors=True)   # bound disk: data is single-use per round
        state = {"completed": r + 1, "final_done": False, "fw": fw}
        _save_state()

    # final fine-tune only once every round is done (not when a deadline stopped us mid-curriculum)
    if state["completed"] >= a.rounds and not state["final_done"]:
        if past_deadline():
            print("[rounds] past deadline -- skipping final fine-tune; resume to run it.", flush=True)
        else:
            print("\n[rounds] ===== FINAL FINE-TUNE  progress=1.0 (final prior, no reshaping) =====",
                  flush=True)
            final_dir = os.path.join(a.data_root, "final")
            gen(final_dir, 1.0, a.seed + 999_000)
            train(final_dir, a.final_steps, {}, a.seed + 999)
            if not a.keep_data:
                shutil.rmtree(final_dir, ignore_errors=True)
            state["final_done"] = True
            _save_state()
    print(f"[rounds] done. total {(time.time()-t0)/3600:.1f}h  "
          f"({state['completed']}/{a.rounds} rounds, final_done={state['final_done']})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate bundle shards (synmac)")
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--progress", type=float, default=0.0, help="curriculum progress in [0,1]")
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--with-artifacts", action="store_true")
    g.set_defaults(func=cmd_gen)

    t = sub.add_parser("train", help="train ModelV3 (synmodel)")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--wd", type=float, default=0.05)
    t.add_argument("--warmup", type=int, default=2000)
    t.add_argument("--k-slots", dest="k_slots", type=int, default=K_SLOTS)
    t.add_argument("--n-global", dest="n_global", type=int, default=N_GLOBAL)
    t.add_argument("--m-max", dest="m_max", type=int, default=None)
    t.add_argument("--val-frac", dest="val_frac", type=float, default=0.1)
    t.add_argument("--eval-every", dest="eval_every", type=int, default=1000)
    t.add_argument("--ckpt-every", dest="ckpt_every", type=int, default=1000)
    t.add_argument("--log-every", dest="log_every", type=int, default=50)
    t.add_argument("--num-workers", dest="num_workers", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu")
    t.add_argument("--resume", default=None, help="checkpoint to continue from (weights+opt+step)")
    t.add_argument("--total-steps", dest="total_steps", type=int, default=None,
                   help="grand total for the p_teacher/w_seed + LR schedules (default: --steps)")
    t.add_argument("--family-weights", dest="family_weights", default=None,
                   help="JSON dict upweighting hard families, e.g. '{\"mu\":2,\"geom\":1.5}'")
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("rounds", help="round-based curriculum training orchestrator (plan Phase 6)")
    r.add_argument("--out", required=True, help="run dir (model + logs persist here across rounds)")
    r.add_argument("--data-root", dest="data_root", required=True, help="where per-round data goes")
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--samples-per-round", dest="spr", type=int, default=2000)
    r.add_argument("--steps-per-round", dest="spr_steps", type=int, default=6000)
    r.add_argument("--final-steps", dest="final_steps", type=int, default=6000)
    r.add_argument("--batch-size", dest="batch_size", type=int, default=4)
    r.add_argument("--gen-workers", dest="gen_workers", type=int, default=14)
    r.add_argument("--num-workers", dest="num_workers", type=int, default=4)
    r.add_argument("--device", default="cuda")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--hard-focus", dest="hard_focus", type=float, default=1.0,
                   help="strength of per-family upweighting from validation error (0 = off)")
    r.add_argument("--max-hours", dest="max_hours", type=float, default=None,
                   help="wall-clock deadline: stop launching new rounds past this many hours")
    r.add_argument("--keep-data", dest="keep_data", action="store_true",
                   help="keep each round's generated data (default: delete after training to save disk)")
    r.add_argument("--synmac-python",
                   default="/mnt/Vol_b/miniconda3/envs/synmac/bin/python")
    r.add_argument("--synmodel-python",
                   default="/mnt/Vol_b/miniconda3/envs/synmodel/bin/python")
    r.set_defaults(func=cmd_rounds)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
