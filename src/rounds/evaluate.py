"""Section 6: per-parameter evaluation on a fixed set (val/test/stress). Runs under synmodel
(needs torch to run the model; matplotlib added there for plotting -- see commit message).

CLI: `evaluate.py --run-dir D --config C --round r --set {val,test,stress}`
"""
import _pathfix  # noqa: F401

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from data import ShardedThetaImageDataset  # noqa: E402
from eval_paths import eval_set_dir, require_not_test_path  # noqa: E402
from model import ParamNet, _load_ckpt  # noqa: E402
from param_space import ParamSpace  # noqa: E402
from train import checkpoint_path, load_config  # noqa: E402


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    MU, LV, Y = [], [], []
    for x, y in loader:
        mu, lv = model(x.to(device))
        MU.append(mu.cpu().numpy())
        LV.append(lv.cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(MU), np.concatenate(LV), np.concatenate(Y)


def _bootstrap_ci(values, n_boot, ci, rng):
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return float(np.quantile(means, lo_q)), float(np.quantile(means, hi_q))


def per_parameter_stats(mu, logvar, y, mask, prior_var, bins=10, n_boot=200, ci=0.95, seed=0):
    """mu/logvar/y/mask: (N, P) arrays. `y` is the normalised [0,1] target (not logit-transformed
    -- this function does that internally, matching ParamNet.to_logit's own eps clamp).

    Returns a dict with one entry per path INDEX (0..P-1); `evaluate_round` attaches path names.
    `mask` gates every reduction (Phase 1: always all-True, but a per-sample mask is accepted so
    Phase 2's real masking is a config/composition change here too, not a new function).
    """
    N, P = y.shape
    eps = ParamNet.EPS
    yc = np.clip(y, eps, 1 - eps)
    z_logit = np.log(yc) - np.log1p(-yc)
    p_med = 1.0 / (1.0 + np.exp(-mu))  # posterior median in [0,1] (sigmoid preserves quantiles)
    se = (p_med - y) ** 2  # (N, P), squared error in [0,1] units

    sig_z = np.exp(0.5 * logvar)  # sd in logit units
    sig01 = sig_z * p_med * (1.0 - p_med)  # delta method -> sd in [0,1] units, per sample
    z = (z_logit - mu) / sig_z  # ~N(0,1) iff calibrated

    nll_elem = 0.5 * (logvar + (z_logit - mu) ** 2 * np.exp(-logvar))

    n_valid = mask.sum(axis=0)
    n_valid_safe = np.clip(n_valid, 1, None)

    def masked_mean(a):
        return (a * mask).sum(axis=0) / n_valid_safe

    mse = masked_mean(se)
    nmse = mse / prior_var
    nll = masked_mean(nll_elem)
    z_mean = masked_mean(z)
    z_var = masked_mean((z - z_mean) ** 2)
    z_std = np.sqrt(z_var)
    cov68 = masked_mean((np.abs(z) <= 1.0).astype(float))
    cov95 = masked_mean((np.abs(z) <= 1.96).astype(float))
    mean_sigma2 = masked_mean(sig01 ** 2)  # mean(sigma^2), per spec -- not (mean sigma)^2
    info_gain = 1.0 - mean_sigma2 / prior_var

    rng = np.random.default_rng(seed)
    nmse_ci_lo = np.empty(P)
    nmse_ci_hi = np.empty(P)
    for k in range(P):
        vals = se[mask[:, k].astype(bool), k] / prior_var[k]
        nmse_ci_lo[k], nmse_ci_hi[k] = _bootstrap_ci(vals, n_boot, ci, rng)

    bin_idx = np.clip((y * bins).astype(int), 0, bins - 1)
    binned_mse = np.full((P, bins), np.nan)
    binned_count = np.zeros((P, bins), dtype=int)
    for k in range(P):
        valid_k = mask[:, k].astype(bool)
        for b in range(bins):
            sel = valid_k & (bin_idx[:, k] == b)
            binned_count[k, b] = int(sel.sum())
            if sel.any():
                binned_mse[k, b] = se[sel, k].mean()

    return dict(
        n_valid=n_valid.astype(int), mse=mse, nmse=nmse, nmse_ci_lo=nmse_ci_lo,
        nmse_ci_hi=nmse_ci_hi, nll=nll, z_mean=z_mean, z_std=z_std, cov68=cov68, cov95=cov95,
        info_gain=info_gain, binned_mse=binned_mse, binned_count=binned_count,
        mu=mu, logvar=logvar, y=y,
    )


def _plot_nmse_across_rounds(eval_dir, paths, out_path):
    files = sorted(glob.glob(os.path.join(eval_dir, "round_*.json")))
    if not files:
        return
    rounds, nmse_by_round = [], []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        rounds.append(d["round"])
        nmse_by_round.append([d["per_parameter"][p]["nmse"] for p in paths])
    arr = np.array(nmse_by_round)  # (n_rounds, P)
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in range(arr.shape[1]):
        ax.plot(rounds, arr[:, k], color="tab:blue", alpha=0.15, linewidth=0.8)
    ax.plot(rounds, np.median(arr, axis=1), color="black", linewidth=2, label="median")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="nMSE=1 (no better than prior)")
    ax.set_xlabel("round")
    ax.set_ylabel("nMSE")
    ax.set_title("nMSE per parameter across rounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_binned_heatmap(binned_nmse, paths, out_path, max_rows=60):
    order = np.argsort(-np.nanmean(binned_nmse, axis=1))[:max_rows]
    fig, ax = plt.subplots(figsize=(6, max(4, 0.18 * len(order))))
    im = ax.imshow(binned_nmse[order], aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([paths[i] for i in order], fontsize=5)
    ax.set_xlabel("bin")
    ax.set_title(f"binned nMSE (worst {len(order)} of {len(paths)} params by mean)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_scatter_grid(mu, y, paths, nmse, out_path, max_params=24):
    order = np.argsort(-nmse)[:max_params]
    n = len(order)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    p_med = 1.0 / (1.0 + np.exp(-mu))
    for i, k in enumerate(order):
        ax = axes[i // ncols][i % ncols]
        ax.scatter(y[:, k], p_med[:, k], s=4, alpha=0.4)
        ax.plot([0, 1], [0, 1], color="red", linewidth=1)
        ax.set_title(f"{paths[k]}\nnMSE={nmse[k]:.2f}", fontsize=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    fig.suptitle(f"predicted vs true (worst {n} of {len(paths)} params by nMSE)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate_round(run_dir, cfg_dict, round_idx, set_kind="val", ckpt_round=None, device=None,
                   n_boot=200, bins=None):
    if set_kind == "test":
        # the guard below also catches this, but fail with the clearest possible message first
        raise PermissionError(
            "evaluate_round must not be called with set_kind='test' -- only finalize may read "
            "the test set"
        )
    eval_cfg = cfg_dict.get("proposal", {})
    bins = bins if bins is not None else int(eval_cfg.get("bins", 10))

    ps = ParamSpace.load(os.path.join(run_dir, "param_space.json"))
    prior_var = np.full(len(ps), ps.prior_var())

    set_dir = eval_set_dir(run_dir, set_kind)
    require_not_test_path(set_dir, "evaluate")
    ds = ShardedThetaImageDataset(set_dir, augment=False)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader = torch.utils.data.DataLoader(ds, batch_size=64)

    ckpt_round = ckpt_round if ckpt_round is not None else round_idx
    ck = _load_ckpt(checkpoint_path(run_dir, ckpt_round), device)
    model = ParamNet(ck["in_ch"], ck["n_out"], width=ck["width"], depth=ck["depth"]).to(device)
    model.load_state_dict(ck["model"])

    mu, logvar, y = _predict(model, loader, device)
    mask_row = ps.validity_mask().astype(np.float64)
    mask = np.broadcast_to(mask_row, y.shape)

    stats = per_parameter_stats(mu, logvar, y, mask, prior_var, bins=bins, n_boot=n_boot,
                                seed=cfg_dict["run"]["master_seed"])

    per_parameter = {}
    for k, path in enumerate(ps.paths):
        per_parameter[path] = dict(
            group=ps.groups[k], n_valid=int(stats["n_valid"][k]), mse=float(stats["mse"][k]),
            nmse=float(stats["nmse"][k]), nmse_ci=[float(stats["nmse_ci_lo"][k]),
                                                   float(stats["nmse_ci_hi"][k])],
            nll=float(stats["nll"][k]), z_mean=float(stats["z_mean"][k]),
            z_std=float(stats["z_std"][k]), coverage68=float(stats["cov68"][k]),
            coverage95=float(stats["cov95"][k]), info_gain=float(stats["info_gain"][k]),
            binned_mse=[None if np.isnan(v) else float(v) for v in stats["binned_mse"][k]],
            binned_count=stats["binned_count"][k].tolist(),
        )

    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    report = dict(round=round_idx, set=set_kind, ckpt_round=ckpt_round, n_samples=len(ds),
                 bins=bins, n_boot=n_boot, per_parameter=per_parameter)
    with open(os.path.join(eval_dir, f"round_{round_idx:03d}.json"), "w") as f:
        json.dump(report, f, indent=1)

    csv_path = os.path.join(eval_dir, f"round_{round_idx:03d}.csv")
    with open(csv_path, "w") as f:
        f.write("path,group,n_valid,nmse,nmse_ci_lo,nmse_ci_hi,nll,z_mean,z_std,coverage68,"
               "coverage95,info_gain\n")
        for path, d in per_parameter.items():
            f.write(f"{path},{d['group']},{d['n_valid']},{d['nmse']:.6g},"
                   f"{d['nmse_ci'][0]:.6g},{d['nmse_ci'][1]:.6g},{d['nll']:.6g},"
                   f"{d['z_mean']:.6g},{d['z_std']:.6g},{d['coverage68']:.4g},"
                   f"{d['coverage95']:.4g},{d['info_gain']:.6g}\n")

    plots_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_nmse_across_rounds(eval_dir, ps.paths,
                             os.path.join(plots_dir, f"nmse_across_rounds_upto_{round_idx:03d}.png"))
    _plot_binned_heatmap(stats["binned_mse"] / prior_var[:, None], ps.paths,
                        os.path.join(plots_dir, f"binned_heatmap_round_{round_idx:03d}.png"))
    _plot_scatter_grid(mu, y, ps.paths, stats["nmse"],
                       os.path.join(plots_dir, f"scatter_round_{round_idx:03d}.png"))

    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--set", default="val", choices=["val", "test", "stress"], dest="set_kind")
    ap.add_argument("--ckpt-round", type=int, default=None)
    a = ap.parse_args()

    cfg_dict = load_config(a.config)
    report = evaluate_round(a.run_dir, cfg_dict, a.round, set_kind=a.set_kind,
                            ckpt_round=a.ckpt_round)
    med = float(np.median([d["nmse"] for d in report["per_parameter"].values()]))
    print(f"round {a.round}: median nMSE = {med:.4f} over {len(report['per_parameter'])} params")


if __name__ == "__main__":
    main()
