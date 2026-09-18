"""Fixture builders for train.py's tests -- fabricate shard files directly (matching the format
generate.py writes) rather than going through the real/mock generator, since train.py's tests run
under synmodel (no scipy) and the generator lives in synmac. stdlib+numpy only."""
import json
import os

import numpy as np

from param_space import ParamSpace


def make_param_space(n_params=6):
    paths = [f"p{i}" for i in range(n_params)]
    return ParamSpace(paths=paths, groups=paths, lo=[0.0] * n_params, hi=[1.0] * n_params,
                      tf=["linear"] * n_params)


def write_shard(out_dir, shard_idx, n, ps, n_markers, seed=0, h=16, w=16, u_fn=None):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    P = len(ps)
    u = (u_fn(rng, n, P) if u_fn else rng.random((n, P))).astype(np.float32)
    # channel c's mean level linear in u[c] (mirrors generate.py's mock renderer), so a model
    # trained on this fixture has real signal to learn, not pure noise.
    levels = 500.0 + 3000.0 * u[:, np.arange(n_markers) % P]
    img = rng.poisson(lam=levels[:, None, None, :].repeat(h, 1).repeat(w, 2))
    img = np.clip(img, 0, 65535).astype(np.uint16)
    stem = os.path.join(out_dir, f"shard_{shard_idx:04d}")
    with open(stem + ".images.npy", "wb") as f:
        np.save(f, img)
    with open(stem + ".npz", "wb") as f:
        np.savez_compressed(f, u=u, component_id=np.zeros(n, dtype=np.int32),
                            n_constant_channels=np.zeros(n, dtype=np.int32))


def make_run_dir(run_dir, ps, n_markers, n_rounds=1, n_per_round=12, n_val=8, seed=0):
    os.makedirs(run_dir, exist_ok=True)
    ps.save(os.path.join(run_dir, "param_space.json"))
    with open(os.path.join(run_dir, "composition.json"), "w") as f:
        json.dump(dict(n_markers=n_markers, n_types=3, expr_frac=0.6, struct_seed=seed,
                       with_artifacts=True, max_aggregates=None, max_fussels=None,
                       max_detach_cover=None), f)
    for r in range(n_rounds):
        round_dir = os.path.join(run_dir, "rounds", f"round_{r:03d}")
        write_shard(round_dir, 0, n_per_round, ps, n_markers, seed=seed + r)
        with open(os.path.join(round_dir, "proposal_used.json"), "w") as f:
            json.dump(dict(round=r, alpha_prior=1.0, bins=5, widen=False,
                          components=[dict(kind="prior", weight=1.0)]), f)
        with open(os.path.join(round_dir, "round_meta.json"), "w") as f:
            json.dump(dict(round=r, valid_count=n_per_round), f)
    val_dir = os.path.join(run_dir, "eval_sets", "val")
    write_shard(val_dir, 0, n_val, ps, n_markers, seed=seed + 1000)
    return run_dir
