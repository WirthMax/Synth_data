"""Shared test fixtures: a tiny config dict and the mock-renderer env toggle, reused by
generate/train/evaluate/policy tests so none of them pay real render cost."""
import os


def make_tiny_config(run_dir=None, master_seed=0, n_markers=4, n_types=3, struct_seed=1,
                     workers=3, shard_size=4, render_timeout_s=30, max_extra_attempts_frac=0.3,
                     store_masks=False, max_retries=3, with_artifacts=True):
    return {
        "run": {"master_seed": master_seed},
        "generation": {
            "workers": workers, "shard_size": shard_size, "render_timeout_s": render_timeout_s,
            "max_extra_attempts_frac": max_extra_attempts_frac, "store_masks": store_masks,
            "max_retries": max_retries,
        },
        "phase1": {
            "composition": {
                "n_markers": n_markers, "n_types": n_types, "expr_frac": 0.6,
                "struct_seed": struct_seed, "with_artifacts": with_artifacts,
                "max_aggregates": None, "max_fussels": None, "max_detach_cover": None,
            },
            "n0": shard_size, "n_per_round": shard_size,
        },
        "proposal": {"alpha_prior": 0.5, "bins": 5, "widen": False},
        "policy": {"top_m": 2, "nmse_floor": 0.05, "delta_min": 0.02, "patience": 2,
                  "max_ratio": 3.0, "min_bin_count": 5, "prior_mix": 0.3, "ema": 0.5,
                  "exclude": []},
        "train": {"steps_per_round": 50, "early_stopping_patience": 2, "beta_nll": None,
                 "max_bad_steps": 20},
        "eval_sets": {"val_n": 40, "test_n": 40, "stress_n": 0},
    }


class MockRendererEnv:
    """Context manager: `with MockRendererEnv(monkeypatch, fail_indices=(3,), sleep_s=0.1):`"""

    def __init__(self, monkeypatch, fail_indices=(), sleep_s=0.0):
        self.monkeypatch = monkeypatch
        self.fail_indices = fail_indices
        self.sleep_s = sleep_s

    def __enter__(self):
        self.monkeypatch.setenv("ROUNDS_MOCK_RENDERER", "1")
        self.monkeypatch.setenv("ROUNDS_MOCK_FAIL_INDICES",
                                ",".join(str(i) for i in self.fail_indices))
        self.monkeypatch.setenv("ROUNDS_MOCK_SLEEP_S", str(self.sleep_s))
        return self

    def __exit__(self, *exc):
        for k in ("ROUNDS_MOCK_RENDERER", "ROUNDS_MOCK_FAIL_INDICES", "ROUNDS_MOCK_SLEEP_S"):
            self.monkeypatch.delenv(k, raising=False)


def repo_src_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
