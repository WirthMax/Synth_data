import numpy as np
import pytest

from eval_sets import make_eval_set, require_not_test_path, NAMESPACE
from generate import generate_round
from proposal import Proposal
from helpers import MockRendererEnv, make_tiny_config


def test_val_test_stress_use_disjoint_seed_namespaces_and_never_collide(tmp_path, monkeypatch):
    with MockRendererEnv(monkeypatch):
        cfg = make_tiny_config(shard_size=4)
        run_dir = tmp_path / "run"
        val_meta = make_eval_set(str(run_dir), cfg, "val", n=6)
        test_meta = make_eval_set(str(run_dir), cfg, "test", n=6)

        val_theta = np.concatenate([np.load(run_dir / "eval_sets/val" / f)["u"]
                                    for f in ("shard_0000.npz", "shard_0001.npz")])
        test_theta = np.concatenate([np.load(run_dir / "eval_sets/test" / f)["u"]
                                     for f in ("shard_0000.npz", "shard_0001.npz")])
        assert not np.array_equal(val_theta, test_theta)
        assert val_meta["valid_count"] == 6
        assert test_meta["valid_count"] == 6


def test_eval_sets_and_training_rounds_share_the_same_param_space(tmp_path, monkeypatch):
    with MockRendererEnv(monkeypatch):
        cfg = make_tiny_config(shard_size=4)
        run_dir = tmp_path / "run"
        generate_round(str(run_dir), cfg, 0, 4, Proposal.prior_only(0))
        val_meta = make_eval_set(str(run_dir), cfg, "val", n=4)
        assert val_meta["n_fitted_params"] > 0
        # a mismatched composition must be rejected, exactly as it is between two rounds
        cfg2 = make_tiny_config(n_markers=99, n_types=99, shard_size=4)
        with pytest.raises(RuntimeError, match="ParamSpace layout mismatch"):
            make_eval_set(str(run_dir), cfg2, "test", n=4)


def test_full_composition_mode_not_implemented(tmp_path, monkeypatch):
    with MockRendererEnv(monkeypatch):
        cfg = make_tiny_config()
        with pytest.raises(NotImplementedError):
            make_eval_set(str(tmp_path / "run"), cfg, "val", n=4, composition_mode="full")


def test_require_not_test_path_blocks_test_set_allows_others(tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(PermissionError):
        require_not_test_path(str(run_dir / "eval_sets" / "test" / "shard_0000.npz"), "train")
    with pytest.raises(PermissionError):
        require_not_test_path(str(run_dir / "eval_sets" / "test"), "policy")
    # val/stress and ordinary round paths must NOT be blocked
    require_not_test_path(str(run_dir / "eval_sets" / "val" / "shard_0000.npz"), "train")
    require_not_test_path(str(run_dir / "rounds" / "round_000" / "manifest.csv"), "train")
    # a coincidental "test" substring elsewhere in the path must not false-positive
    require_not_test_path(str(run_dir / "latest_test_run" / "manifest.csv"), "train")


def test_namespace_ids_are_disjoint_and_outside_plausible_round_range():
    values = list(NAMESPACE.values())
    assert len(set(values)) == len(values)
    assert all(v > 100_000 for v in values), "namespace sentinels must be far outside round ids"
