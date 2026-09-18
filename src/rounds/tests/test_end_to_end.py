"""The end-to-end smoke test from the spec's Tests section: mock renderer, tiny model, a few
rounds + finalize, all on CPU, under 2 minutes -- and the deliberately no-effect theta dimension
(generate.py's mock renderer never reads the LAST index of u for any channel, by construction)
must end up marked saturated rather than chased forever.

Uses REAL subprocess calls into both conda envs (generate.py/eval_sets.py need synmac's scipy;
train.py/evaluate.py/finalize.py need synmodel's torch) -- exactly what Orchestrator does in
production, just against ROUNDS_MOCK_RENDERER=1 instead of the real renderer. Runs under either
env (orchestrate.py itself needs only numpy+yaml), skipped if either conda env isn't present on
this machine.
"""
import json
import os
import subprocess

import pytest
import yaml

from orchestrate import Orchestrator

SYNMAC_PY = "/mnt/Vol_b/miniconda3/envs/synmac/bin/python"
SYNMODEL_PY = "/mnt/Vol_b/miniconda3/envs/synmodel/bin/python"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(SYNMAC_PY) and os.path.exists(SYNMODEL_PY)),
    reason="needs both the synmac and synmodel conda envs on this machine",
)


def _e2e_config(n_params_hint=6):
    # top_m generously covers every real candidate so the no-effect parameter is targeted every
    # round it's a candidate, removing "did it get picked" randomness from what this test
    # verifies -- the interesting, non-deterministic-in-principle question is whether its
    # measured non-improvement reliably clears the saturation threshold, not whether the policy
    # happens to rank it in the top few out of dozens.
    return {
        "run": {"master_seed": 0, "max_rounds": 3, "synmac_python": SYNMAC_PY,
               "synmodel_python": SYNMODEL_PY},
        "generation": {"workers": 2, "shard_size": 8, "render_timeout_s": 30,
                      "max_extra_attempts_frac": 0.5, "store_masks": False, "max_retries": 2},
        "phase1": {"composition": {"n_markers": 4, "n_types": 3, "expr_frac": 0.6,
                                   "struct_seed": 1, "with_artifacts": True,
                                   "max_aggregates": None, "max_fussels": None,
                                   "max_detach_cover": None},
                  "n0": 16, "n_per_round": 16},
        "proposal": {"alpha_prior": 0.5, "bins": 5, "widen": False},
        "policy": {"top_m": 500, "nmse_floor": 0.0, "delta_min": 0.02, "patience": 2,
                  "max_ratio": 3.0, "min_bin_count": 3, "prior_mix": 0.3, "ema": 0.5,
                  "exclude": []},
        "train": {"steps_per_round": 60, "warmup_steps": 10, "eval_every_steps": 20,
                 "early_stopping_patience": 5, "max_bad_steps": 30, "batch": 8, "lr": 2e-3,
                 "width": 8, "depth": 2},
        "eval_sets": {"val_n": 24, "test_n": 24, "stress_n": 0},
        "finalize": {"steps": 30, "batch": 8, "lr": 5e-4},
    }


def test_end_to_end_three_rounds_and_no_effect_param_saturates(tmp_path):
    run_dir = str(tmp_path / "run")
    config_path = tmp_path / "config.yaml"
    cfg = _e2e_config()
    config_path.write_text(yaml.safe_dump(cfg))

    old_environ = dict(os.environ)
    os.environ["ROUNDS_MOCK_RENDERER"] = "1"
    try:
        # make_eval_set/val+test sets, via the real synmac subprocess, mock-rendered
        rounds_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for kind in ("val", "test"):
            subprocess.run([SYNMAC_PY, os.path.join(rounds_dir, "eval_sets.py"),
                           "--run-dir", run_dir, "--config", str(config_path),
                           "--kind", kind], check=True, env=os.environ)

        orch = Orchestrator(run_dir, cfg, allow_dirty=True, lag=0)
        orch.run(config_path=str(config_path))

        # all 3 rounds must have produced a checkpoint and an eval report
        for r in range(3):
            assert os.path.exists(os.path.join(run_dir, "checkpoints", f"round_{r:03d}.pt"))
            assert os.path.exists(os.path.join(run_dir, "eval", f"round_{r:03d}.json"))
        assert os.path.exists(os.path.join(run_dir, "summary.md"))

        with open(os.path.join(run_dir, "param_space.json")) as f:
            ps_paths = json.load(f)["paths"]
        no_effect_param = ps_paths[-1]  # never read by the mock renderer, by construction

        with open(os.path.join(run_dir, "policy_state.json")) as f:
            saturation_state = json.load(f)
        assert saturation_state.get(no_effect_param, {}).get("saturated") is True, (
            f"expected {no_effect_param!r} (the mock renderer's deliberately unlearnable "
            f"parameter) to be marked saturated after 3 rounds; got "
            f"{saturation_state.get(no_effect_param)}"
        )

        # finalize (importance mode), the one place allowed to read the test set
        subprocess.run([SYNMODEL_PY, os.path.join(rounds_dir, "finalize.py"),
                       "--run-dir", run_dir, "--config", str(config_path),
                       "--mode", "importance"], check=True, env=os.environ)
        assert os.path.exists(os.path.join(run_dir, "final_report.md"))
        assert os.path.exists(os.path.join(run_dir, "checkpoints", "final.pt"))
    finally:
        os.environ.clear()
        os.environ.update(old_environ)
