"""generate_round tests, all against the fast mock renderer (ROUNDS_MOCK_RENDERER=1) -- these
exercise the orchestration (seeding, sharding, resumability, failure handling, disk check), not
the real generator, which is covered separately by the (slow, opt-in) real-render smoke test."""
import csv
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from generate import generate_round
from proposal import Proposal
from helpers import MockRendererEnv, make_tiny_config, repo_src_dir


def test_seed_determinism_independent_of_worker_count(tmp_path, monkeypatch):
    with MockRendererEnv(monkeypatch):
        cfg_a = make_tiny_config(workers=1, shard_size=3)
        cfg_b = make_tiny_config(workers=4, shard_size=3)
        run_a, run_b = tmp_path / "a", tmp_path / "b"
        generate_round(str(run_a), cfg_a, 0, 6, Proposal.prior_only(0))
        generate_round(str(run_b), cfg_b, 0, 6, Proposal.prior_only(0))

        for shard in ("shard_0000", "shard_0001"):
            img_a = np.load(run_a / "rounds/round_000" / f"{shard}.images.npy")
            img_b = np.load(run_b / "rounds/round_000" / f"{shard}.images.npy")
            assert np.array_equal(img_a, img_b)
            u_a = np.load(run_a / "rounds/round_000" / f"{shard}.npz")["u"]
            u_b = np.load(run_b / "rounds/round_000" / f"{shard}.npz")["u"]
            assert np.array_equal(u_a, u_b)


def test_resumability_skips_completed_shards_and_produces_no_duplicates(tmp_path, monkeypatch):
    with MockRendererEnv(monkeypatch):
        cfg = make_tiny_config(workers=2, shard_size=3)
        run_dir = tmp_path / "run"
        generate_round(str(run_dir), cfg, 0, 6, Proposal.prior_only(0))
        before = np.load(run_dir / "rounds/round_000/shard_0000.images.npy")
        mtime_before = os.path.getmtime(run_dir / "rounds/round_000/shard_0000.images.npy")

        meta = generate_round(str(run_dir), cfg, 0, 9, Proposal.prior_only(0))

        after = np.load(run_dir / "rounds/round_000/shard_0000.images.npy")
        mtime_after = os.path.getmtime(run_dir / "rounds/round_000/shard_0000.images.npy")
        assert np.array_equal(before, after)
        assert mtime_before == mtime_after  # not even rewritten
        assert meta["valid_count"] == 9

        indices = []
        with open(run_dir / "rounds/round_000/manifest.csv") as f:
            for row in csv.DictReader(f):
                indices.append(int(row["index"]))
        assert len(indices) == len(set(indices)), "resumed round produced duplicate indices"
        assert sorted(indices) == list(range(9))


def test_failures_are_logged_and_topped_up_from_new_indices(tmp_path, monkeypatch):
    with MockRendererEnv(monkeypatch, fail_indices=(1, 4)):
        cfg = make_tiny_config(workers=2, shard_size=3, max_extra_attempts_frac=1.0,
                               max_retries=1)
        run_dir = tmp_path / "run"
        meta = generate_round(str(run_dir), cfg, 0, 6, Proposal.prior_only(0))
        assert meta["valid_count"] == 6
        assert meta["extra_attempts_used"] == 2

        failures = [json.loads(line) for line in
                   open(run_dir / "rounds/round_000/failures.jsonl")]
        assert {f["index"] for f in failures} == {1, 4}

        indices = []
        with open(run_dir / "rounds/round_000/manifest.csv") as f:
            for row in csv.DictReader(f):
                indices.append(int(row["index"]))
        assert 1 not in indices and 4 not in indices
        assert len(indices) == 6


def test_render_timeout_counts_as_a_failure(tmp_path, monkeypatch):
    # sleep (2s) comfortably exceeds render_timeout_s (1s) -- SIGALRM should fire and turn this
    # into a failure rather than a hang. With max_extra_attempts_frac=0.0 every index gets
    # exactly one doomed attempt and none can succeed, so the round must abort with a clear
    # message rather than hang or silently return 0 samples -- this is what actually proves the
    # timeout fired (a forced-failure-by-index test alone never exercises SIGALRM at all).
    with MockRendererEnv(monkeypatch, sleep_s=2.0):
        cfg = make_tiny_config(workers=2, shard_size=2, max_extra_attempts_frac=0.0,
                               max_retries=1, render_timeout_s=1)
        with pytest.raises(RuntimeError, match="max_extra_attempts_frac"):
            generate_round(str(tmp_path / "run"), cfg, 0, 2, Proposal.prior_only(0))
        failures = [json.loads(line) for line in
                   open(tmp_path / "run" / "rounds/round_000/failures.jsonl")]
        assert any("RenderTimeout" in f["error"] for f in failures)


def test_exceeding_max_extra_attempts_aborts_with_clear_message(tmp_path, monkeypatch):
    with MockRendererEnv(monkeypatch, fail_indices=(0, 1, 2, 3, 4, 5)):
        cfg = make_tiny_config(workers=2, shard_size=3, max_extra_attempts_frac=0.2,
                               max_retries=1)
        run_dir = tmp_path / "run"
        with pytest.raises(RuntimeError, match="max_extra_attempts_frac"):
            generate_round(str(run_dir), cfg, 0, 6, Proposal.prior_only(0))


def test_disk_space_precondition_checked_before_generating(tmp_path, monkeypatch):
    with MockRendererEnv(monkeypatch):
        cfg = make_tiny_config()
        called = {}

        def _fake_check(*a, **k):
            called["yes"] = True
            raise RuntimeError("not enough free disk space (test)")

        import generate as generate_mod
        monkeypatch.setattr(generate_mod, "_check_disk_space", _fake_check)
        with pytest.raises(RuntimeError, match="not enough free disk space"):
            generate_round(str(tmp_path / "run"), cfg, 0, 6, Proposal.prior_only(0))
        assert called.get("yes")


def test_param_space_mismatch_across_rounds_raises(tmp_path, monkeypatch):
    with MockRendererEnv(monkeypatch):
        cfg1 = make_tiny_config(n_markers=4, n_types=3)
        run_dir = tmp_path / "run"
        generate_round(str(run_dir), cfg1, 0, 3, Proposal.prior_only(0))

        cfg2 = make_tiny_config(n_markers=6, n_types=5)  # different composition, same run dir
        with pytest.raises(RuntimeError, match="ParamSpace layout mismatch"):
            generate_round(str(run_dir), cfg2, 1, 3, Proposal.prior_only(1))


def test_process_hygiene_no_orphans_after_sigkill(tmp_path):
    psutil = pytest.importorskip("psutil")
    driver = os.path.join(os.path.dirname(__file__), "_hygiene_driver.py")
    env = dict(os.environ, ROUNDS_MOCK_RENDERER="1", ROUNDS_MOCK_SLEEP_S="1.0")
    proc = subprocess.Popen([sys.executable, driver, str(tmp_path / "run")], env=env,
                            cwd=repo_src_dir())
    try:
        parent = psutil.Process(proc.pid)
        # Give the driver time to spawn its worker pool AND for each worker to get through
        # generate.py's own (numpy/scipy/numba) import chain and reach _worker_init's prctl
        # call -- a worker killed during that bootstrap window, before prctl has registered,
        # cannot have PDEATHSIG armed yet. children() becoming non-empty only proves the OS
        # process exists, not that its own init has run, so wait for a fixed settle period on
        # top of that rather than treating "children visible" as "ready".
        deadline = time.time() + 20
        children = []
        while time.time() < deadline:
            children = parent.children(recursive=True)
            if children:
                break
            time.sleep(0.2)
        assert children, "driver never spawned worker processes"
        time.sleep(3.0)
        children = parent.children(recursive=True) or children

        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)

        deadline = time.time() + 5
        alive = [c for c in children if c.is_running()]
        while time.time() < deadline and alive:
            time.sleep(0.2)
            alive = [c for c in children if c.is_running()]
        assert not alive, f"{len(alive)} worker process(es) survived parent SIGKILL"
    finally:
        if proc.poll() is None:
            proc.kill()
