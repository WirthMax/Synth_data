"""Section 5: train_round -- warm-started, step-based training on the union of rounds 0..r.
Runs under `synmodel` (torch; deliberately no scipy) -- every import here must stay safe there.

CLI: `train.py --run-dir D --config C --upto r [--from-scratch]`
"""
import _pathfix  # noqa: F401

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import yaml

from model import ParamNet, build_model, _load_ckpt
from data import ShardedThetaImageDataset
from param_space import ParamSpace
from composition import CompositionConfig
from eval_paths import eval_set_dir, require_not_test_path
from provenance import config_hash, git_commit


def checkpoint_path(run_dir, r):
    return os.path.join(run_dir, "checkpoints", f"round_{r:03d}.pt")


def masked_mse_warmup(mu, y01, mask):
    z = ParamNet.to_logit(y01)
    se = (mu - z) ** 2 * mask
    return se.sum() / mask.sum().clamp_min(1.0)


def masked_nll(mu, logvar, y01, mask, beta=None, sample_weights=None):
    """Masked Gaussian NLL in logit space (ParamNet.nll's own math, generalised with a mask and
    two optional, both-off-by-default knobs):

    - beta-NLL (Seitzer et al., ICLR 2022): weight each element by its own (detached) predictive
      variance^beta, which stops the loss from underweighting high-error/high-variance elements
      relative to plain NLL's implicit 1/sigma^2 down-weighting.
    - sample_weights: per-sample importance weights (Section 9's finalize, off during rounds per
      spec), broadcast across the parameter dimension before the mask is applied.
    """
    z = ParamNet.to_logit(y01)
    nll = 0.5 * (logvar + (z - mu) ** 2 * torch.exp(-logvar))
    if beta:
        nll = nll * (torch.exp(logvar).detach() ** beta)
    nll = nll * mask
    if sample_weights is not None:
        w = sample_weights.view(-1, 1)
        return (nll * w).sum() / (mask * w).sum().clamp_min(1e-12)
    return nll.sum() / mask.sum().clamp_min(1.0)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def _val_nll(model, loader, device, mask_dev):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mu, logvar = model(x)
        mb = mask_dev.unsqueeze(0).expand_as(mu)
        total += float(masked_nll(mu, logvar, y, mb)) * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


def _load_json_if_exists(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _arch_kwargs_equal(a, b):
    """Dict equality that's array-aware -- `set_transformer`'s arch_kwargs carry numpy index
    arrays (marker_scatter/global_path_idx), which `!=` can't compare directly."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if not np.array_equal(np.asarray(va), np.asarray(vb)):
                return False
        elif va != vb:
            return False
    return True


def _resolve_arch(train_cfg, ps, in_ch, image_size):
    """(arch, arch_kwargs) for building a fresh model. `train.width`/`train.depth` stay the
    top-level knobs for `arch: cnn` (unchanged from before this existed); `arch: set_transformer`
    reads its hyperparameters from the new `train.arch_kwargs` sub-dict, plus the
    marker/global reassembly tables derived once from `ps` here."""
    arch = train_cfg.get("arch", "cnn")
    if arch == "cnn":
        return arch, dict(width=int(train_cfg.get("width", 32)),
                          depth=int(train_cfg.get("depth", 4)))
    if arch == "set_transformer":
        mfm = ps.marker_field_map(in_ch)
        kwargs = dict(train_cfg.get("arch_kwargs", {}))
        kwargs.setdefault("img_size", image_size)
        kwargs["marker_scatter"] = mfm["marker_scatter"]
        kwargs["global_path_idx"] = mfm["global_path_idx"]
        return arch, kwargs
    raise ValueError(f"unknown train.arch {arch!r} (expected 'cnn' or 'set_transformer')")


def train_round(run_dir, cfg_dict, upto_round, from_scratch=False, device=None, repo_dir=None):
    train_cfg = cfg_dict["train"]
    comp = CompositionConfig.from_dict(cfg_dict["phase1"]["composition"])
    ps = ParamSpace.load(os.path.join(run_dir, "param_space.json"))
    mask = torch.from_numpy(ps.validity_mask().astype(np.float32))

    round_dirs = []
    for r in range(upto_round + 1):
        d = os.path.join(run_dir, "rounds", f"round_{r:03d}")
        require_not_test_path(d, "train_round")
        round_dirs.append(d)
    augment_seed = int(cfg_dict["run"]["master_seed"])
    train_ds = ShardedThetaImageDataset(round_dirs, augment=True, augment_seed=augment_seed)

    val_dir = eval_set_dir(run_dir, "val")
    require_not_test_path(val_dir, "train_round")
    val_ds = ShardedThetaImageDataset(val_dir, augment=False)

    print(f"train_round: {len(train_ds)} train samples (rounds 0..{upto_round}), "
         f"{len(val_ds)} val samples, {len(ps)} parameters", flush=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch = int(train_cfg.get("batch", 16))
    dl_tr = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)
    dl_va = torch.utils.data.DataLoader(val_ds, batch_size=batch)

    in_ch, n_out = comp.n_markers, len(ps)
    arch, arch_kwargs = _resolve_arch(train_cfg, ps, in_ch, train_ds.image_size)
    model = build_model(arch, in_ch, n_out, **arch_kwargs).to(device)

    warm_started_from = None
    prev_path = checkpoint_path(run_dir, upto_round - 1)
    if not from_scratch and upto_round > 0 and os.path.exists(prev_path):
        ck = _load_ckpt(prev_path, device)
        prev_arch = ck.get("arch", "cnn")
        prev_kwargs = ck.get("arch_kwargs", dict(width=ck.get("width"), depth=ck.get("depth")))
        if ((ck["in_ch"], ck["n_out"], prev_arch) != (in_ch, n_out, arch)
                or not _arch_kwargs_equal(prev_kwargs, arch_kwargs)):
            raise RuntimeError(
                f"cannot warm-start round {upto_round} from {prev_path}: architecture mismatch "
                f"(in_ch/n_out/arch {ck['in_ch']}/{ck['n_out']}/{prev_arch} -> "
                f"{in_ch}/{n_out}/{arch}, arch_kwargs {prev_kwargs} -> {arch_kwargs}) -- "
                f"composition and train.arch/train.width/train.depth/train.arch_kwargs must not "
                f"change within a run"
            )
        model.load_state_dict(ck["model"])
        warm_started_from = upto_round - 1

    lr = float(train_cfg.get("lr", 3e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = int(train_cfg["steps_per_round"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    warmup_steps = int(train_cfg.get("warmup_steps", max(1, steps // 10)))
    eval_every = max(1, int(train_cfg.get("eval_every_steps", max(1, steps // 10))))
    patience = int(train_cfg.get("early_stopping_patience", 3))
    max_bad_steps = int(train_cfg.get("max_bad_steps", 20))
    beta_nll = train_cfg.get("beta_nll")

    mask_dev = mask.to(device)
    best_val_nll = math.inf
    best_state, best_opt_state, best_sched_state, best_step = None, None, None, 0
    bad_evals, bad_steps_streak = 0, 0
    log_path = os.path.join(run_dir, "rounds", f"round_{upto_round:03d}", "train_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    t0 = time.time()
    train_iter = _cycle(dl_tr)
    step = 0
    with open(log_path, "a") as logf:
        while step < steps:
            model.train()
            x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            mu, logvar = model(x)
            mb = mask_dev.unsqueeze(0).expand_as(mu)
            loss = (masked_mse_warmup(mu, y, mb) if step < warmup_steps
                   else masked_nll(mu, logvar, y, mb, beta=beta_nll))

            if not torch.isfinite(loss):
                bad_steps_streak += 1
                logf.write(json.dumps(dict(step=step, event="non_finite_loss",
                                           streak=bad_steps_streak)) + "\n")
                if bad_steps_streak > max_bad_steps:
                    raise RuntimeError(
                        f"train_round: {bad_steps_streak} consecutive non-finite losses at step "
                        f"{step}, aborting (max_bad_steps={max_bad_steps})"
                    )
                step += 1
                continue
            bad_steps_streak = 0

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % eval_every == 0 or step == steps:
                val_nll = _val_nll(model, dl_va, device, mask_dev)
                improved = val_nll < best_val_nll - 1e-4
                logf.write(json.dumps(dict(step=step, train_loss=float(loss), val_nll=val_nll,
                                           improved=improved,
                                           elapsed_s=time.time() - t0)) + "\n")
                logf.flush()
                print(f"  step {step:6d}/{steps}  train_loss {float(loss):8.4f}  "
                     f"val_nll {val_nll:8.4f}{'  *' if improved else ''}", flush=True)
                if improved:
                    best_val_nll = val_nll
                    best_step = step
                    best_state = {k: v.detach().cpu().clone() for k, v in
                                 model.state_dict().items()}
                    best_opt_state = opt.state_dict()
                    best_sched_state = sched.state_dict()
                    bad_evals = 0
                else:
                    bad_evals += 1
                if bad_evals >= patience:
                    print(f"  early stopping: no val improvement for {patience} checks",
                         flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    round_dir = os.path.join(run_dir, "rounds", f"round_{upto_round:03d}")
    proposal_used = _load_json_if_exists(os.path.join(round_dir, "proposal_used.json"))
    saturation_state = _load_json_if_exists(os.path.join(run_dir, "policy_state.json"))

    ckpt = dict(
        model=best_state if best_state is not None else model.state_dict(),
        optimizer=best_opt_state, scheduler=best_sched_state,
        step=best_step, round=upto_round, rounds_trained_on=list(range(upto_round + 1)),
        config_hash=config_hash(cfg_dict),
        git_commit=git_commit(repo_dir or os.path.dirname(os.path.abspath(__file__))),
        proposal=proposal_used, saturation_state=saturation_state,
        in_ch=in_ch, n_out=n_out, arch=arch, arch_kwargs=arch_kwargs, paths=ps.paths,
        lo=ps.lo, hi=ps.hi, best_val_nll=best_val_nll, warm_started_from=warm_started_from,
    )
    os.makedirs(os.path.dirname(checkpoint_path(run_dir, upto_round)), exist_ok=True)
    tmp = checkpoint_path(run_dir, upto_round) + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, checkpoint_path(run_dir, upto_round))

    meta = dict(round=upto_round, n_train=len(train_ds), n_val=len(val_ds),
               best_val_nll=best_val_nll, best_step=best_step, total_steps=step,
               warm_started_from=warm_started_from, elapsed_s=time.time() - t0)
    with open(os.path.join(round_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--upto", type=int, required=True)
    ap.add_argument("--from-scratch", action="store_true", dest="from_scratch")
    a = ap.parse_args()

    cfg_dict = load_config(a.config)
    meta = train_round(a.run_dir, cfg_dict, a.upto, from_scratch=a.from_scratch)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
