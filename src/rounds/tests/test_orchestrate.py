"""orchestrate.py tests. The real generate/train/evaluate steps are subprocess calls into other
envs -- these tests fake those four methods to exercise the loop's own logic (state tracking,
resumability, stopping conditions, summary generation) without needing either env's heavy deps.
"""
import json
import os

import pytest

from orchestrate import Orchestrator


def _tiny_config(**overrides):
    cfg = dict(
        run={"master_seed": 0, "max_rounds": 3},
        phase1={"composition": {"n_markers": 4, "n_types": 3}, "n0": 10, "n_per_round": 10},
        proposal={"alpha_prior": 0.5, "bins": 5, "widen": False},
        policy={"top_m": 1, "nmse_floor": 0.05, "delta_min": 0.02, "patience": 2,
               "max_ratio": 3.0, "min_bin_count": 5, "prior_mix": 0.3, "ema": 0.5, "exclude": []},
    )
    cfg["run"].update(overrides.pop("run", {}))
    cfg.update(overrides)
    return cfg


class FakeSteps:
    """Mixin recording calls and fabricating just enough on-disk state for should_stop/
    update_summary/_step_done to behave like the real steps did."""

    def __init__(self, *a, fail_on=None, **k):
        super().__init__(*a, **k)
        self.calls = []
        self.fail_on = fail_on or set()

    def generate(self, round_idx, config_path, n=None, background=False):
        self.calls.append(("generate", round_idx))
        if ("generate", round_idx) in self.fail_on:
            raise RuntimeError("fake generate failure")
        round_dir = os.path.join(self.run_dir, "rounds", f"round_{round_idx:03d}")
        os.makedirs(round_dir, exist_ok=True)
        with open(os.path.join(round_dir, "round_meta.json"), "w") as f:
            json.dump(dict(round=round_idx, valid_count=n or 10), f)

        class _FakeProc:
            def wait(self_inner):
                return 0

            def poll(self_inner):
                return 0

        return _FakeProc()

    def train(self, round_idx, config_path):
        self.calls.append(("train", round_idx))

    def evaluate(self, round_idx, config_path, set_kind="val"):
        self.calls.append(("evaluate", round_idx))
        eval_dir = os.path.join(self.run_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, f"round_{round_idx:03d}.json"), "w") as f:
            json.dump(dict(round=round_idx, per_parameter={"a": {"nmse": 0.5}}), f)

    def decide_next_proposal(self):
        self.calls.append(("policy", None))
        decisions_dir = os.path.join(self.run_dir, "decisions")
        os.makedirs(decisions_dir, exist_ok=True)
        n_done = len([c for c in self.calls if c[0] == "policy"])
        with open(os.path.join(decisions_dir, f"round_{n_done - 1:03d}.json"), "w") as f:
            json.dump(dict(picks=[]), f)


class FakeOrchestrator(FakeSteps, Orchestrator):
    pass


def _write_config_file(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("run:\n  master_seed: 0\n")
    return str(p)


def test_full_run_executes_all_steps_in_order_for_each_round(tmp_path):
    cfg = _tiny_config(run={"master_seed": 0, "max_rounds": 2})
    orch = FakeOrchestrator(str(tmp_path / "run"), cfg, allow_dirty=True)
    orch.run(config_path=_write_config_file(tmp_path))
    steps = [c[0] for c in orch.calls]
    assert steps == ["generate", "train", "evaluate", "policy"] * 2


def test_resumability_skips_completed_steps_on_rerun(tmp_path):
    run_dir = str(tmp_path / "run")
    cfg = _tiny_config(run={"master_seed": 0, "max_rounds": 1})
    orch1 = FakeOrchestrator(run_dir, cfg, allow_dirty=True)
    # simulate a crash right after "train" completes: mark generate+train done, then stop
    state = orch1.load_state()
    orch1.generate(0, "cfg.yaml")
    orch1._mark_step_done(state, 0, "generate")
    orch1.train(0, "cfg.yaml")
    orch1._mark_step_done(state, 0, "train")
    orch1.calls.clear()

    orch2 = FakeOrchestrator(run_dir, cfg, allow_dirty=True)
    orch2.run(config_path=_write_config_file(tmp_path))
    steps = [c[0] for c in orch2.calls]
    assert steps == ["evaluate", "policy"]  # generate/train NOT repeated


def test_max_rounds_stops_the_loop(tmp_path):
    cfg = _tiny_config(run={"master_seed": 0, "max_rounds": 2})
    orch = FakeOrchestrator(str(tmp_path / "run"), cfg, allow_dirty=True)
    orch.run(config_path=_write_config_file(tmp_path))
    rounds_touched = sorted({c[1] for c in orch.calls if c[1] is not None})
    assert rounds_touched == [0, 1]


def test_max_total_frames_stops_the_loop(tmp_path):
    cfg = _tiny_config(run={"master_seed": 0, "max_rounds": 100, "max_total_frames": 15})
    orch = FakeOrchestrator(str(tmp_path / "run"), cfg, allow_dirty=True)
    orch.run(config_path=_write_config_file(tmp_path))
    # n0=10 for round 0, n_per_round=10 for round 1 -> total hits 20 >= 15 stop check happens
    # BEFORE round 1 starts (10 < 15) so round 1 still runs, then round 2 is blocked
    rounds_touched = sorted({c[1] for c in orch.calls if c[1] is not None})
    assert rounds_touched == [0, 1]


def test_sigint_style_stop_leaves_no_partial_step_marked_done(tmp_path):
    run_dir = str(tmp_path / "run")
    cfg = _tiny_config(run={"master_seed": 0, "max_rounds": 5})
    orch = FakeOrchestrator(run_dir, cfg, allow_dirty=True, fail_on=set())

    real_train = orch.train

    def _train_then_stop(round_idx, config_path):
        real_train(round_idx, config_path)
        orch._stop_requested = True

    orch.train = _train_then_stop
    orch.run(config_path=_write_config_file(tmp_path))
    state = orch.load_state()
    assert [0, "generate"] in state["completed_steps"]
    assert [0, "train"] in state["completed_steps"]
    assert [0, "evaluate"] not in state["completed_steps"]


def test_summary_md_written_with_a_row_per_round(tmp_path):
    cfg = _tiny_config(run={"master_seed": 0, "max_rounds": 2})
    orch = FakeOrchestrator(str(tmp_path / "run"), cfg, allow_dirty=True)
    orch.run(config_path=_write_config_file(tmp_path))
    summary_path = os.path.join(orch.run_dir, "summary.md")
    assert os.path.exists(summary_path)
    text = open(summary_path).read()
    assert "| 0 |" in text and "| 1 |" in text


def test_run_never_calls_evaluate_with_test_set():
    import inspect
    src = inspect.getsource(Orchestrator)
    assert '"test"' not in src.replace("set_kind=\"val\"", "")


def test_config_snapshot_written_once(tmp_path):
    cfg = _tiny_config(run={"master_seed": 0, "max_rounds": 1})
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run:\n  master_seed: 0\n")
    orch = FakeOrchestrator(str(tmp_path / "run"), cfg, allow_dirty=True)
    orch.run(config_path=str(config_path))
    snap = os.path.join(orch.run_dir, "config_snapshot.yaml")
    assert os.path.exists(snap)


def test_lag_pipelining_generates_next_round_in_background(tmp_path):
    cfg = _tiny_config(run={"master_seed": 0, "max_rounds": 3})
    orch = FakeOrchestrator(str(tmp_path / "run"), cfg, allow_dirty=True, lag=1)
    orch.run(config_path=_write_config_file(tmp_path))
    gen_rounds = [c[1] for c in orch.calls if c[0] == "generate"]
    # every round's generate should still have happened exactly once each, just pipelined
    assert sorted(gen_rounds) == [0, 1, 2]


def test_allow_dirty_false_raises_on_dirty_tree(tmp_path, monkeypatch):
    import orchestrate as orch_mod

    def _fake_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = " M some_file.py\n"
        return R()

    monkeypatch.setattr(orch_mod.subprocess, "run", _fake_run)
    cfg = _tiny_config()
    orch = FakeOrchestrator(str(tmp_path / "run"), cfg, allow_dirty=False)
    with pytest.raises(RuntimeError, match="dirty working tree"):
        orch.run(config_path=_write_config_file(tmp_path))
