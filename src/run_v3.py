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

N_GLOBAL = 72          # pad length for the variable-length ARCH globals (max ~65 at 3 structures)
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
    gen = torch.Generator().manual_seed(a.seed)
    logf = open(os.path.join(a.out, "train_log.jsonl"), "a")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ModelV3: {n_params/1e6:.1f}M params | device={device}", flush=True)

    step, t0 = 0, time.time()
    while step < a.steps:
        for tile, present, tgt in train_loader:
            lr = _cosine_lr(step, a.steps, a.lr, a.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            log = TR.train_step(model, opt, tile, present, tgt, step, a.steps, gen, device=device)
            if step % a.log_every == 0:
                log["step"], log["lr"] = step, lr
                print(f"step {step:6d}  loss {log['loss/total']:9.2f}  lr {lr:.2e}  "
                      f"({(time.time()-t0)/(step+1):.2f}s/step)", flush=True)
                logf.write(json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                       for k, v in log.items()}) + "\n")
                logf.flush()
            if step > 0 and step % a.eval_every == 0:
                _run_eval(model, val_loader, device, step, logf)
            if step > 0 and step % a.ckpt_every == 0:
                _save(model, a, step)
            step += 1
            if step >= a.steps:
                break
    _save(model, a, step)
    _run_eval(model, val_loader, device, step, logf)
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


def _save(model, a, step):
    import torch
    path = os.path.join(a.out, "model_last.pt")
    torch.save(dict(model=model.state_dict(), step=step,
                    cfg=dict(n_global=a.n_global, k_slots=a.k_slots)), path)


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
    t.set_defaults(func=cmd_train)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
