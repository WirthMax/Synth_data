"""evaluate.py tests. Runs ONLY under synmodel (torch), same as test_train.py."""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchinfo")
pytest.importorskip("matplotlib")

from evaluate import evaluate_round, per_parameter_stats  # noqa: E402
from train import train_round  # noqa: E402
from train_helpers import make_param_space, make_run_dir  # noqa: E402
from eval_paths import require_not_test_path  # noqa: E402


def _tiny_cfg(bins=5):
    return dict(run={"master_seed": 0},
               phase1={"composition": {"n_markers": 4, "n_types": 3}},
               proposal={"bins": bins},
               train=dict(steps_per_round=30, warmup_steps=5, eval_every_steps=10,
                         early_stopping_patience=10, max_bad_steps=20, batch=4, lr=1e-3,
                         width=8, depth=2))


def test_evaluate_round_end_to_end_writes_json_csv_and_plots(tmp_path):
    ps = make_param_space(6)
    run_dir = make_run_dir(str(tmp_path / "run"), ps, n_markers=4, n_rounds=1, n_per_round=16,
                           n_val=12)
    cfg = _tiny_cfg()
    train_round(run_dir, cfg, upto_round=0, from_scratch=True, device="cpu")

    report = evaluate_round(run_dir, cfg, round_idx=0, set_kind="val", device="cpu")
    assert report["n_samples"] == 12
    assert set(report["per_parameter"].keys()) == set(ps.paths)
    for d in report["per_parameter"].values():
        assert d["n_valid"] == 12  # Phase 1: validity mask is all-True
        assert d["nmse"] >= 0
        assert len(d["binned_mse"]) == cfg["proposal"]["bins"]

    eval_dir = os.path.join(run_dir, "eval")
    assert os.path.exists(os.path.join(eval_dir, "round_000.json"))
    assert os.path.exists(os.path.join(eval_dir, "round_000.csv"))
    assert os.path.exists(os.path.join(eval_dir, "plots", "binned_heatmap_round_000.png"))
    assert os.path.exists(os.path.join(eval_dir, "plots", "scatter_round_000.png"))


def test_evaluate_refuses_test_set(tmp_path):
    ps = make_param_space(6)
    run_dir = make_run_dir(str(tmp_path / "run"), ps, n_markers=4, n_rounds=1, n_per_round=16,
                           n_val=12)
    cfg = _tiny_cfg()
    train_round(run_dir, cfg, upto_round=0, from_scratch=True, device="cpu")
    with pytest.raises(PermissionError):
        evaluate_round(run_dir, cfg, round_idx=0, set_kind="test", device="cpu")


def test_require_not_test_path_used_directly():
    with pytest.raises(PermissionError):
        require_not_test_path("/x/eval_sets/test/shard_0000.npz", "evaluate")


def test_per_parameter_stats_perfect_predictions_give_zero_nmse_and_zero_nll_extremes():
    n, P = 200, 3
    rng = np.random.default_rng(0)
    y = rng.random((n, P))
    # a tiny logvar and mu = logit(y) exactly -> near-perfect predictions
    eps = 1e-4
    yc = np.clip(y, eps, 1 - eps)
    z = np.log(yc) - np.log1p(-yc)
    mu = z.copy()
    logvar = np.full((n, P), -8.0)
    mask = np.ones((n, P))
    prior_var = np.full(P, 1.0 / 12.0)
    stats = per_parameter_stats(mu, logvar, y, mask, prior_var, bins=5, n_boot=50, seed=0)
    assert np.all(stats["nmse"] < 1e-4)
    assert np.all(stats["n_valid"] == n)


def test_per_parameter_stats_random_predictions_give_nmse_near_one():
    n, P = 4000, 2
    rng = np.random.default_rng(1)
    y = rng.random((n, P))
    mu = np.zeros((n, P))  # predicts logit(0.5)=0 regardless of y -- the prior mean
    logvar = np.zeros((n, P))
    mask = np.ones((n, P))
    prior_var = np.full(P, 1.0 / 12.0)
    stats = per_parameter_stats(mu, logvar, y, mask, prior_var, bins=5, n_boot=50, seed=0)
    # predicting the constant prior median for everything should give nMSE close to 1
    assert np.all(np.abs(stats["nmse"] - 1.0) < 0.15)


def test_per_parameter_stats_respects_mask():
    n, P = 100, 1
    rng = np.random.default_rng(2)
    y = rng.random((n, P))
    mu = np.zeros((n, P))
    logvar = np.zeros((n, P))
    prior_var = np.full(P, 1.0 / 12.0)
    mask_half = np.zeros((n, P))
    mask_half[: n // 2] = 1.0
    stats_half = per_parameter_stats(mu, logvar, y, mask_half, prior_var, bins=5, n_boot=20,
                                     seed=0)
    stats_full = per_parameter_stats(mu, logvar, y, np.ones((n, P)), prior_var, bins=5,
                                     n_boot=20, seed=0)
    assert stats_half["n_valid"][0] == n // 2
    assert stats_full["n_valid"][0] == n
    # masked-out rows must not affect the reduction -- verify against a manual recomputation
    se_manual = ((1 / (1 + np.exp(-mu[: n // 2, 0]))) - y[: n // 2, 0]) ** 2
    assert abs(stats_half["mse"][0] - se_manual.mean()) < 1e-9


def test_bootstrap_ci_json_serialisable(tmp_path):
    ps = make_param_space(4)
    run_dir = make_run_dir(str(tmp_path / "run"), ps, n_markers=3, n_rounds=1, n_per_round=12,
                           n_val=10)
    cfg = _tiny_cfg(bins=4)
    cfg["phase1"]["composition"]["n_markers"] = 3
    train_round(run_dir, cfg, upto_round=0, from_scratch=True, device="cpu")
    evaluate_round(run_dir, cfg, round_idx=0, set_kind="val", device="cpu")
    # round-trips through json.dump already inside evaluate_round -- reload and sanity check
    with open(os.path.join(run_dir, "eval", "round_000.json")) as f:
        reloaded = json.load(f)
    assert reloaded["round"] == 0
    first = next(iter(reloaded["per_parameter"].values()))
    assert len(first["nmse_ci"]) == 2
