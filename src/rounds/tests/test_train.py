"""train_round tests. Runs ONLY under synmodel (torch) -- skipped entirely under synmac via
importorskip, since synmac deliberately has no torch. Run with:
  /mnt/Vol_b/miniconda3/envs/synmodel/bin/python -m pytest rounds/tests/test_train.py
(the rest of rounds/tests/ needs scipy and runs under synmac instead -- see the project's
existing two-env split in CLAUDE.md.)
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchinfo")  # model.py imports this; synmac has torch but not torchinfo

from train import train_round, checkpoint_path, masked_nll  # noqa: E402
from train_helpers import make_param_space, make_run_dir  # noqa: E402
from eval_paths import require_not_test_path  # noqa: E402


def _tiny_train_cfg(**overrides):
    cfg = dict(steps_per_round=40, warmup_steps=5, eval_every_steps=10,
              early_stopping_patience=10, max_bad_steps=20, batch=4, lr=1e-3, width=8, depth=2)
    cfg.update(overrides)
    return dict(run={"master_seed": 0},
               phase1={"composition": {"n_markers": 4, "n_types": 3}},
               train=cfg)


def test_train_round_from_scratch_runs_and_writes_checkpoint(tmp_path):
    ps = make_param_space(6)
    run_dir = make_run_dir(str(tmp_path / "run"), ps, n_markers=4, n_rounds=1, n_per_round=16,
                           n_val=8)
    cfg = _tiny_train_cfg()
    meta = train_round(run_dir, cfg, upto_round=0, from_scratch=True, device="cpu")
    assert meta["n_train"] == 16
    assert meta["n_val"] == 8
    assert os.path.exists(checkpoint_path(run_dir, 0))
    ck = torch.load(checkpoint_path(run_dir, 0), weights_only=False)
    assert ck["in_ch"] == 4
    assert ck["n_out"] == len(ps)
    assert ck["round"] == 0
    assert ck["rounds_trained_on"] == [0]
    assert ck["warm_started_from"] is None
    assert ck["proposal"]["round"] == 0


def test_warm_start_loads_previous_round_weights(tmp_path):
    ps = make_param_space(6)
    run_dir = make_run_dir(str(tmp_path / "run"), ps, n_markers=4, n_rounds=2, n_per_round=16,
                           n_val=8)
    cfg = _tiny_train_cfg()
    train_round(run_dir, cfg, upto_round=0, from_scratch=True, device="cpu")
    ck0 = torch.load(checkpoint_path(run_dir, 0), weights_only=False)

    meta1 = train_round(run_dir, cfg, upto_round=1, from_scratch=False, device="cpu")
    assert meta1["warm_started_from"] == 0
    ck1 = torch.load(checkpoint_path(run_dir, 1), weights_only=False)
    assert ck1["rounds_trained_on"] == [0, 1]
    assert ck1["warm_started_from"] == 0
    # optimizer/scheduler are NOT warm-started (fresh each round, per design decision) --
    # only the model weights carry over as the starting point.
    assert ck0["model"].keys() == ck1["model"].keys()


def test_from_scratch_flag_ignores_previous_checkpoint(tmp_path):
    ps = make_param_space(6)
    run_dir = make_run_dir(str(tmp_path / "run"), ps, n_markers=4, n_rounds=2, n_per_round=16,
                           n_val=8)
    cfg = _tiny_train_cfg()
    train_round(run_dir, cfg, upto_round=0, from_scratch=True, device="cpu")
    meta1 = train_round(run_dir, cfg, upto_round=1, from_scratch=True, device="cpu")
    assert meta1["warm_started_from"] is None


def test_architecture_mismatch_on_warm_start_raises(tmp_path):
    ps = make_param_space(6)
    run_dir = make_run_dir(str(tmp_path / "run"), ps, n_markers=4, n_rounds=2, n_per_round=16,
                           n_val=8)
    cfg = _tiny_train_cfg()
    train_round(run_dir, cfg, upto_round=0, from_scratch=True, device="cpu")
    cfg2 = _tiny_train_cfg(width=16)  # different architecture
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        train_round(run_dir, cfg2, upto_round=1, from_scratch=False, device="cpu")


def test_masked_nll_ignores_masked_out_entries():
    mu = torch.zeros(2, 3)
    logvar = torch.zeros(2, 3)
    y = torch.tensor([[0.5, 0.3, 0.5], [0.5, 0.7, 0.5]])  # col 2 has zero loss either way
    mask_all = torch.ones(2, 3)
    mask_partial = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    # corrupt the masked-out column with a value that would change the loss if NOT ignored
    y_corrupt = y.clone()
    y_corrupt[:, 2] = 0.999999
    loss_clean = masked_nll(mu, logvar, y, mask_partial)
    loss_corrupt = masked_nll(mu, logvar, y_corrupt, mask_partial)
    assert torch.allclose(loss_clean, loss_corrupt, atol=1e-4)
    loss_full_mask = masked_nll(mu, logvar, y_corrupt, mask_all)
    assert not torch.allclose(loss_full_mask, loss_clean)


def test_masked_nll_sample_weights_reduce_to_plain_when_uniform():
    mu = torch.randn(4, 3)
    logvar = torch.zeros(4, 3)
    y = torch.rand(4, 3)
    mask = torch.ones(4, 3)
    plain = masked_nll(mu, logvar, y, mask)
    weighted = masked_nll(mu, logvar, y, mask, sample_weights=torch.ones(4))
    assert torch.allclose(plain, weighted, atol=1e-5)


def test_beta_nll_changes_loss_value():
    mu = torch.zeros(4, 3)
    logvar = torch.full((4, 3), 0.5)
    y = torch.rand(4, 3)
    mask = torch.ones(4, 3)
    plain = masked_nll(mu, logvar, y, mask, beta=None)
    beta = masked_nll(mu, logvar, y, mask, beta=1.0)
    assert not torch.allclose(plain, beta)


def test_non_finite_loss_aborts_after_max_bad_steps(tmp_path, monkeypatch):
    ps = make_param_space(6)
    run_dir = make_run_dir(str(tmp_path / "run"), ps, n_markers=4, n_rounds=1, n_per_round=16,
                           n_val=8)
    cfg = _tiny_train_cfg(max_bad_steps=2)

    import train as train_mod

    def _always_nan(*a, **k):
        return torch.tensor(float("nan"))

    monkeypatch.setattr(train_mod, "masked_mse_warmup", _always_nan)
    with pytest.raises(RuntimeError, match="non-finite losses"):
        train_round(run_dir, cfg, upto_round=0, from_scratch=True, device="cpu")


def test_train_round_refuses_a_test_set_path(tmp_path):
    with pytest.raises(PermissionError):
        require_not_test_path(str(tmp_path / "eval_sets" / "test" / "shard_0000.npz"),
                              "train_round")


def test_checkpoint_written_atomically_no_tmp_left_behind(tmp_path):
    ps = make_param_space(6)
    run_dir = make_run_dir(str(tmp_path / "run"), ps, n_markers=4, n_rounds=1, n_per_round=16,
                           n_val=8)
    cfg = _tiny_train_cfg()
    train_round(run_dir, cfg, upto_round=0, from_scratch=True, device="cpu")
    assert not os.path.exists(checkpoint_path(run_dir, 0) + ".tmp")


def test_train_log_records_step_and_val_nll(tmp_path):
    ps = make_param_space(6)
    run_dir = make_run_dir(str(tmp_path / "run"), ps, n_markers=4, n_rounds=1, n_per_round=16,
                           n_val=8)
    cfg = _tiny_train_cfg()
    train_round(run_dir, cfg, upto_round=0, from_scratch=True, device="cpu")
    log_path = os.path.join(run_dir, "rounds", "round_000", "train_log.jsonl")
    assert os.path.exists(log_path)
    with open(log_path) as f:
        lines = [json.loads(line) for line in f]
    assert any("val_nll" in line for line in lines)
