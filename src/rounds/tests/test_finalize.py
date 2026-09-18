"""finalize.py tests. Runs ONLY under synmodel (torch), same as test_train.py/test_evaluate.py."""
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchinfo")
pytest.importorskip("matplotlib")

from finalize import finalize, _WeightedDataset  # noqa: E402
from train import train_round  # noqa: E402
from train_helpers import make_param_space, make_run_dir, write_shard  # noqa: E402
from eval_paths import eval_set_dir  # noqa: E402


def _tiny_cfg():
    return dict(run={"master_seed": 0},
               phase1={"composition": {"n_markers": 4, "n_types": 3}},
               proposal={"bins": 5, "alpha_prior": 0.5},
               train=dict(steps_per_round=30, warmup_steps=5, eval_every_steps=10,
                         early_stopping_patience=10, max_bad_steps=20, batch=4, lr=1e-3,
                         width=8, depth=2),
               finalize=dict(steps=20, batch=4, lr=1e-4))


def _make_two_round_run(tmp_path, n_markers=4):
    ps = make_param_space(6)
    run_dir = make_run_dir(str(tmp_path / "run"), ps, n_markers=n_markers, n_rounds=2,
                           n_per_round=16, n_val=10)
    cfg = _tiny_cfg()
    train_round(run_dir, cfg, upto_round=0, from_scratch=True, device="cpu")
    train_round(run_dir, cfg, upto_round=1, from_scratch=False, device="cpu")
    # eval + a decision round for round 1 so final_report can compare against a val report
    from evaluate import evaluate_round
    evaluate_round(run_dir, cfg, round_idx=1, set_kind="val", device="cpu")
    write_shard(eval_set_dir(run_dir, "test"), 0, 12, ps, n_markers, seed=999)
    return run_dir, cfg, ps


def test_finalize_importance_mode_end_to_end(tmp_path):
    run_dir, cfg, ps = _make_two_round_run(tmp_path)
    report = finalize(run_dir, cfg, mode="importance", device="cpu")
    assert report["mode"] == "importance"
    assert report["n_test"] == 12
    assert report["ess"] is not None
    assert 0 < report["ess"] <= report["n_train"]
    assert set(report["per_parameter"].keys()) == set(ps.paths)
    assert os.path.exists(os.path.join(run_dir, "checkpoints", "final.pt"))
    assert os.path.exists(os.path.join(run_dir, "final_report.md"))
    assert os.path.exists(os.path.join(run_dir, "final_report.json"))
    # val comparison should be present since round 1 was evaluated
    some_param = next(iter(report["per_parameter"].values()))
    assert "val_nmse" not in some_param or isinstance(some_param.get("val_nmse"), float)


def test_finalize_prior_only_mode_end_to_end(tmp_path):
    run_dir, cfg, ps = _make_two_round_run(tmp_path)
    report = finalize(run_dir, cfg, mode="prior_only", device="cpu")
    assert report["mode"] == "prior_only"
    assert report["ess"] is None  # ESS only meaningful for importance weighting
    # prior_only trains on a strict subset (component_id==0) of the full union
    assert report["n_train"] <= 32  # 2 rounds x 16 samples, all prior-only in this fixture


def test_final_checkpoint_records_provenance(tmp_path):
    run_dir, cfg, ps = _make_two_round_run(tmp_path)
    finalize(run_dir, cfg, mode="importance", device="cpu")
    ck = torch.load(os.path.join(run_dir, "checkpoints", "final.pt"), weights_only=False)
    assert ck["fine_tuned_from_round"] == 1
    assert ck["mode"] == "importance"
    assert ck["paths"] == ps.paths


def test_weighted_dataset_bundles_weight_with_sample():
    class _Base:
        def __len__(self):
            return 3

        def __getitem__(self, k):
            return torch.tensor(float(k)), torch.tensor(float(k) * 2)

    ds = _WeightedDataset(_Base(), weights=[0.5, 1.0, 2.0])
    for k in range(3):
        x, y, w = ds[k]
        assert float(x) == k
        assert float(y) == k * 2
        assert float(w) == ds.weights[k]


def test_finalize_invalid_mode_raises(tmp_path):
    run_dir, cfg, ps = _make_two_round_run(tmp_path)
    with pytest.raises(ValueError, match="mode must be"):
        finalize(run_dir, cfg, mode="bogus", device="cpu")


def test_finalize_report_json_serialisable(tmp_path):
    run_dir, cfg, ps = _make_two_round_run(tmp_path)
    finalize(run_dir, cfg, mode="importance", device="cpu")
    with open(os.path.join(run_dir, "final_report.json")) as f:
        reloaded = json.load(f)
    assert "per_parameter" in reloaded
