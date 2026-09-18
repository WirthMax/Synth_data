"""Section 8: `run_rounds` -- the outer loop (generate -> train -> evaluate -> policy -> next
proposal), resumable at the step level via run_state.json.

generate.py/eval_sets.py need scipy (synmac); train.py/evaluate.py need torch (synmodel) -- no
single env has both, so the orchestrator shells out to each as a subprocess using the interpreter
path configured for that env, rather than importing them in-process. policy.py needs neither and
runs in-process directly. This file itself needs only stdlib + yaml + the (scipy/torch-free)
policy module, so it can run under either env, or a bare one with just those two installed.

CLI: `orchestrate.py --run-dir D --config C`
"""
import _pathfix  # noqa: F401

import argparse
import json
import os
import signal
import subprocess
import time

import yaml

from policy import run_policy
from provenance import git_commit


STEPS = ("generate", "train", "evaluate", "policy")


class StopRequested(Exception):
    pass


class Orchestrator:
    def __init__(self, run_dir, config, allow_dirty=False, lag=0):
        self.run_dir = run_dir
        self.config = config
        self.allow_dirty = allow_dirty
        self.lag = lag
        self._current_proc = None
        self._stop_requested = False
        self._pending_generate = {}  # round -> Popen, for --lag 1 pipelining

    # -- run_state.json (atomic, step-level resumability) ------------------------------------

    def _state_path(self):
        return os.path.join(self.run_dir, "run_state.json")

    def load_state(self):
        p = self._state_path()
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return dict(completed_steps=[], stopped=False)

    def save_state(self, state):
        tmp = self._state_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, self._state_path())

    def _step_done(self, state, round_idx, step):
        return [round_idx, step] in state["completed_steps"]

    def _mark_step_done(self, state, round_idx, step):
        state["completed_steps"].append([round_idx, step])
        self.save_state(state)

    # -- signal handling: stop between steps, never leave a subprocess orphaned --------------

    def install_signal_handlers(self):
        def _handler(signum, frame):
            self._stop_requested = True
            if self._current_proc is not None and self._current_proc.poll() is None:
                self._current_proc.terminate()
                try:
                    self._current_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._current_proc.kill()

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _check_stop(self):
        if self._stop_requested:
            raise StopRequested()

    # -- subprocess helpers -------------------------------------------------------------------

    def _run(self, python, script, args, background=False):
        cmd = [python, script] + [str(a) for a in args]
        if background:
            return subprocess.Popen(cmd)
        self._current_proc = subprocess.Popen(cmd)
        ret = self._current_proc.wait()
        proc, self._current_proc = self._current_proc, None
        if ret != 0:
            raise RuntimeError(f"subprocess failed (exit {ret}): {' '.join(cmd)}")
        return proc

    def _script(self, name):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

    def _synmac_python(self):
        return self.config["run"].get("synmac_python",
                                      "/mnt/Vol_b/miniconda3/envs/synmac/bin/python")

    def _synmodel_python(self):
        return self.config["run"].get("synmodel_python",
                                      "/mnt/Vol_b/miniconda3/envs/synmodel/bin/python")

    # -- per-round steps ----------------------------------------------------------------------

    def _proposal_path(self, round_idx):
        return os.path.join(self.run_dir, "proposals", f"round_{round_idx:03d}.json")

    def generate(self, round_idx, config_path, n=None, background=False):
        args = ["--run-dir", self.run_dir, "--config", config_path, "--round", round_idx]
        if n is not None:
            args += ["--n", n]
        proposal_path = self._proposal_path(round_idx)
        if os.path.exists(proposal_path):
            args += ["--proposal", proposal_path]
        return self._run(self._synmac_python(), self._script("generate.py"), args,
                         background=background)

    def train(self, round_idx, config_path):
        self._run(self._synmodel_python(), self._script("train.py"),
                  ["--run-dir", self.run_dir, "--config", config_path, "--upto", round_idx])

    def evaluate(self, round_idx, config_path, set_kind="val"):
        self._run(self._synmodel_python(), self._script("evaluate.py"),
                  ["--run-dir", self.run_dir, "--config", config_path, "--round", round_idx,
                   "--set", set_kind])

    def decide_next_proposal(self):
        return run_policy(self.run_dir, self.config, dry_run=False)

    # -- stopping conditions --------------------------------------------------------------------

    def _total_frames_so_far(self):
        total = 0
        rounds_dir = os.path.join(self.run_dir, "rounds")
        if not os.path.isdir(rounds_dir):
            return 0
        for name in os.listdir(rounds_dir):
            meta_path = os.path.join(rounds_dir, name, "round_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    total += json.load(f).get("valid_count", 0)
        return total

    def should_stop(self, round_idx, start_time):
        run_cfg = self.config["run"]
        if "max_rounds" in run_cfg and round_idx >= run_cfg["max_rounds"]:
            return f"max_rounds={run_cfg['max_rounds']} reached"
        if "max_total_frames" in run_cfg:
            total = self._total_frames_so_far()
            if total >= run_cfg["max_total_frames"]:
                return f"max_total_frames={run_cfg['max_total_frames']} reached ({total} so far)"
        if "max_wall_hours" in run_cfg:
            elapsed_h = (time.time() - start_time) / 3600.0
            if elapsed_h >= run_cfg["max_wall_hours"]:
                return f"max_wall_hours={run_cfg['max_wall_hours']} reached"
        return None

    # -- summary.md -----------------------------------------------------------------------------

    def update_summary(self):
        lines = ["# Run summary\n", f"config hash: `{_config_hash(self.config)}`\n",
                f"git commit: `{git_commit(os.path.dirname(self._script('')))}`\n",
                "| round | N | picks | median nMSE |", "|---|---|---|---|"]
        rounds_dir = os.path.join(self.run_dir, "rounds")
        eval_dir = os.path.join(self.run_dir, "eval")
        decisions_dir = os.path.join(self.run_dir, "decisions")
        if os.path.isdir(rounds_dir):
            for name in sorted(os.listdir(rounds_dir)):
                meta_path = os.path.join(rounds_dir, name, "round_meta.json")
                if not os.path.exists(meta_path):
                    continue
                with open(meta_path) as f:
                    meta = json.load(f)
                r = meta["round"]
                eval_path = os.path.join(eval_dir, f"round_{r:03d}.json")
                med_nmse = "-"
                if os.path.exists(eval_path):
                    with open(eval_path) as f:
                        report = json.load(f)
                    import statistics
                    med_nmse = f"{statistics.median(d['nmse'] for d in report['per_parameter'].values()):.4f}"
                decision_path = os.path.join(decisions_dir, f"round_{r:03d}.json")
                picks = "-"
                if os.path.exists(decision_path):
                    with open(decision_path) as f:
                        picks = ", ".join(json.load(f)["picks"]) or "(prior only)"
                lines.append(f"| {r} | {meta['valid_count']} | {picks} | {med_nmse} |")
        with open(os.path.join(self.run_dir, "summary.md"), "w") as f:
            f.write("\n".join(lines) + "\n")

    # -- the loop ---------------------------------------------------------------------------

    def run(self, config_path):
        os.makedirs(self.run_dir, exist_ok=True)
        _check_clean_tree(os.path.dirname(self._script("")), self.allow_dirty)
        self._snapshot_config(config_path)

        state = self.load_state()
        start_time = time.time()
        round_idx = 0
        try:
            while True:
                self._check_stop()
                stop_reason = self.should_stop(round_idx, start_time)
                if stop_reason:
                    print(f"stopping: {stop_reason}", flush=True)
                    break

                if not self._step_done(state, round_idx, "generate"):
                    print(f"round {round_idx}: generate", flush=True)
                    if self.lag > 0 and round_idx in self._pending_generate:
                        self._pending_generate.pop(round_idx).wait()
                    else:
                        n = self._n_for_round(round_idx)
                        self.generate(round_idx, config_path, n=n)
                    self._mark_step_done(state, round_idx, "generate")
                self._check_stop()

                lookahead_stop = self.should_stop(round_idx + 1, start_time)
                if self.lag > 0 and (round_idx + 1) not in self._pending_generate \
                        and lookahead_stop is None:
                    next_n = self._n_for_round(round_idx + 1)
                    self._pending_generate[round_idx + 1] = self.generate(
                        round_idx + 1, config_path, n=next_n, background=True)

                if not self._step_done(state, round_idx, "train"):
                    print(f"round {round_idx}: train", flush=True)
                    self.train(round_idx, config_path)
                    self._mark_step_done(state, round_idx, "train")
                self._check_stop()

                if not self._step_done(state, round_idx, "evaluate"):
                    print(f"round {round_idx}: evaluate", flush=True)
                    self.evaluate(round_idx, config_path)
                    self._mark_step_done(state, round_idx, "evaluate")
                self._check_stop()

                if not self._step_done(state, round_idx, "policy"):
                    print(f"round {round_idx}: policy", flush=True)
                    self.decide_next_proposal()
                    self._mark_step_done(state, round_idx, "policy")
                self.update_summary()

                round_idx += 1
        except StopRequested:
            print("stopped by signal", flush=True)
        finally:
            for pending in self._pending_generate.values():
                if pending.poll() is None:
                    pending.terminate()

    def _n_for_round(self, round_idx):
        phase1 = self.config["phase1"]
        return phase1["n0"] if round_idx == 0 else phase1["n_per_round"]

    def _snapshot_config(self, config_path):
        dest = os.path.join(self.run_dir, "config_snapshot.yaml")
        if not os.path.exists(dest):
            with open(config_path) as src, open(dest, "w") as dst:
                dst.write(src.read())


def _config_hash(config):
    from provenance import config_hash
    return config_hash(config)


def _check_clean_tree(repo_hint_dir, allow_dirty):
    if allow_dirty:
        return
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(repo_hint_dir)))
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir,
                             capture_output=True, text=True, timeout=5)
    except Exception:
        return  # not a git repo / git unavailable -- nothing to check
    if out.returncode == 0 and out.stdout.strip():
        raise RuntimeError(
            "refusing to start on a dirty working tree (uncommitted changes) -- commit or "
            "stash first, or pass --allow-dirty if this is intentional"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--lag", type=int, default=0, choices=[0, 1])
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    with open(a.config) as f:
        config = yaml.safe_load(f)
    orch = Orchestrator(a.run_dir, config, allow_dirty=a.allow_dirty, lag=a.lag)
    orch.install_signal_handlers()
    orch.run(a.config)


if __name__ == "__main__":
    main()
