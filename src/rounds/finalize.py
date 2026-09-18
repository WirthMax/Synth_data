"""Section 9: finalize -- fine-tune on the full union of rounds (importance-weighted, or
prior-component-only), then ONE evaluation on the held-out test set. This is the only stage
allowed to open the test set -- it does not call require_not_test_path at all, by design, since
the whole point of finalize is the one sanctioned read.

Runs under synmodel (torch).

CLI: `finalize.py --run-dir D --config C [--mode {importance,prior_only}]`
"""
import _pathfix  # noqa: F401

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

from data import ShardedThetaImageDataset
from eval_paths import eval_set_dir
from evaluate import _predict, per_parameter_stats
from model import ParamNet, _load_ckpt
from param_space import ParamSpace
from proposal import Proposal, RoundRecord, importance_weights
from train import _cycle, checkpoint_path, load_config, masked_nll


class _WeightedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to also return a per-sample weight, bundled INTO each item -- so a
    shuffling DataLoader keeps (x, y, w) aligned per sample automatically, rather than needing to
    track original dataset indices through the shuffle to look a weight up separately."""

    def __init__(self, base, weights):
        self.base = base
        self.weights = weights

    def __len__(self):
        return len(self.base)

    def __getitem__(self, k):
        x, y = self.base[k]
        return x, y, torch.tensor(float(self.weights[k]))


def _round_dirs(run_dir):
    pattern = re.compile(r"round_(\d{3})$")
    dirs = []
    for d in sorted(glob.glob(os.path.join(run_dir, "rounds", "round_*"))):
        m = pattern.search(d)
        if m and os.path.exists(os.path.join(d, "round_meta.json")):
            dirs.append((int(m.group(1)), d))
    dirs.sort()
    return dirs


def _load_round_record(round_dir, ps):
    proposal = Proposal.load(os.path.join(round_dir, "proposal_used.json"))
    thetas = [np.load(p)["u"] for p in sorted(glob.glob(os.path.join(round_dir, "shard_*.npz")))]
    return RoundRecord(proposal=proposal, u=np.concatenate(thetas, axis=0))


def _sample_weights_importance(run_dir, ps, round_dirs):
    records = [_load_round_record(d, ps) for _, d in round_dirs]
    w, ess = importance_weights(ps, records)
    w = w / w.mean()  # normalised to mean 1, per spec
    return w, ess


def finalize(run_dir, cfg_dict, mode="importance", device=None, repo_dir=None):
    if mode not in ("importance", "prior_only"):
        raise ValueError(f"mode must be 'importance' or 'prior_only', got {mode!r}")
    final_cfg = cfg_dict.get("finalize", {})
    ps = ParamSpace.load(os.path.join(run_dir, "param_space.json"))
    prior_var = np.full(len(ps), ps.prior_var())

    round_dirs = _round_dirs(run_dir)
    if not round_dirs:
        raise RuntimeError(f"no completed rounds found under {run_dir}/rounds")
    r_last = round_dirs[-1][0]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dirs_only = [d for _, d in round_dirs]
    full_ds = ShardedThetaImageDataset(dirs_only, augment=True,
                                       augment_seed=cfg_dict["run"]["master_seed"])

    ess = None
    if mode == "importance":
        w, ess = _sample_weights_importance(run_dir, ps, round_dirs)
        if len(w) != len(full_ds):
            raise RuntimeError(
                f"importance weight count ({len(w)}) doesn't match dataset size "
                f"({len(full_ds)}) -- shard read order must have diverged between "
                f"ShardedThetaImageDataset and _load_round_record"
            )
        train_ds = _WeightedDataset(full_ds, w)
    else:
        prior_idx = np.where(full_ds.component_id == 0)[0]
        subset = torch.utils.data.Subset(full_ds, prior_idx.tolist())
        train_ds = _WeightedDataset(subset, np.ones(len(subset)))

    batch = int(final_cfg.get("batch", 16))
    dl = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True)

    ck = _load_ckpt(checkpoint_path(run_dir, r_last), device)
    model = ParamNet(ck["in_ch"], ck["n_out"], width=ck["width"], depth=ck["depth"]).to(device)
    model.load_state_dict(ck["model"])

    steps = int(final_cfg.get("steps", 500))
    lr = float(final_cfg.get("lr", 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    mask_dev = torch.from_numpy(ps.validity_mask().astype(np.float32)).to(device)

    t0 = time.time()
    train_iter = _cycle(dl)
    for step in range(steps):
        model.train()
        x, y, w = next(train_iter)
        x, y, w = x.to(device), y.to(device), w.to(device)
        mu, logvar = model(x)
        mb = mask_dev.unsqueeze(0).expand_as(mu)
        loss = masked_nll(mu, logvar, y, mb, sample_weights=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    final_ckpt_path = os.path.join(run_dir, "checkpoints", "final.pt")
    tmp = final_ckpt_path + ".tmp"
    torch.save(dict(model=model.state_dict(), in_ch=ck["in_ch"], n_out=ck["n_out"],
                    width=ck["width"], depth=ck["depth"], mode=mode, ess=ess,
                    fine_tuned_from_round=r_last, steps=steps, paths=ps.paths), tmp)
    os.replace(tmp, final_ckpt_path)

    # -- the one sanctioned test-set read -----------------------------------------------------
    test_dir = eval_set_dir(run_dir, "test")
    test_ds = ShardedThetaImageDataset(test_dir, augment=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)
    mu, logvar, y = _predict(model, test_loader, device)
    mask = np.broadcast_to(ps.validity_mask().astype(np.float64), y.shape)
    stats = per_parameter_stats(mu, logvar, y, mask, prior_var,
                                bins=int(cfg_dict.get("proposal", {}).get("bins", 10)),
                                n_boot=200, seed=cfg_dict["run"]["master_seed"])

    val_report_path = os.path.join(run_dir, "eval", f"round_{r_last:03d}.json")
    val_report = None
    if os.path.exists(val_report_path):
        with open(val_report_path) as f:
            val_report = json.load(f)

    report = _write_final_report(run_dir, ps, stats, mode, ess, r_last, val_report,
                                 elapsed_s=time.time() - t0, n_test=len(test_ds),
                                 n_train=len(train_ds))
    return report


def _write_final_report(run_dir, ps, stats, mode, ess, r_last, val_report, elapsed_s, n_test,
                        n_train):
    per_parameter = {}
    for k, path in enumerate(ps.paths):
        per_parameter[path] = dict(
            nmse=float(stats["nmse"][k]), nmse_ci=[float(stats["nmse_ci_lo"][k]),
                                                   float(stats["nmse_ci_hi"][k])],
            nll=float(stats["nll"][k]), z_mean=float(stats["z_mean"][k]),
            z_std=float(stats["z_std"][k]), coverage68=float(stats["cov68"][k]),
            coverage95=float(stats["cov95"][k]), info_gain=float(stats["info_gain"][k]),
        )
        if val_report and path in val_report.get("per_parameter", {}):
            per_parameter[path]["val_nmse"] = val_report["per_parameter"][path]["nmse"]

    report = dict(mode=mode, ess=ess, fine_tuned_from_round=r_last, n_train=n_train,
                 n_test=n_test, elapsed_s=elapsed_s, per_parameter=per_parameter)
    with open(os.path.join(run_dir, "final_report.json"), "w") as f:
        json.dump(report, f, indent=1)

    lines = ["# Final report", "", f"mode: `{mode}`", f"fine-tuned from round: {r_last}",
            f"train samples: {n_train}  test samples: {n_test}"]
    if ess is not None:
        lines.append(f"effective sample size (importance weights): {ess:.1f} / {n_train} "
                    f"({100 * ess / n_train:.1f}%)")
    med_test = float(np.median(stats["nmse"]))
    lines.append(f"\nmedian test nMSE: {med_test:.4f}")
    if val_report:
        med_val = float(np.median([d["nmse"] for d in val_report["per_parameter"].values()]))
        lines.append(f"median val nMSE (round {r_last}, for comparison): {med_val:.4f}")
    lines.append("\n| parameter | test nMSE | CI | val nMSE | NLL | z_mean | z_std | "
                "cov68 | cov95 | info gain |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for path, d in sorted(per_parameter.items(), key=lambda kv: -kv[1]["nmse"]):
        val_nmse = d.get("val_nmse")
        val_str = f"{val_nmse:.4f}" if val_nmse is not None else "-"
        lines.append(f"| {path} | {d['nmse']:.4f} | "
                    f"[{d['nmse_ci'][0]:.4f},{d['nmse_ci'][1]:.4f}] | {val_str} | "
                    f"{d['nll']:.4f} | {d['z_mean']:.3f} | {d['z_std']:.3f} | "
                    f"{d['coverage68']:.3f} | {d['coverage95']:.3f} | {d['info_gain']:.3f} |")
    with open(os.path.join(run_dir, "final_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="importance", choices=["importance", "prior_only"])
    a = ap.parse_args()
    cfg_dict = load_config(a.config)
    report = finalize(a.run_dir, cfg_dict, mode=a.mode)
    print(f"median test nMSE: {np.median([d['nmse'] for d in report['per_parameter'].values()]):.4f}")


if __name__ == "__main__":
    main()
