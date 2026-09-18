"""Section 3: `generate_round` -- render one round's frames from a proposal, in parallel, with
resumable shard writes, failure logging, and a disk-space precondition.

CLI: `generate.py --run-dir D --config C --round r [--n N] [--proposal P.json]`
"""
import _pathfix  # noqa: F401

import argparse
import csv
import ctypes
import json
import math
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import yaml

import config as cfg
from gen_dataset import render_frame
import render as render_mod

from composition import CompositionConfig, build_struct, build_param_space
from param_space import ParamSpace
from proposal import Proposal, sample_theta
from provenance import config_hash, git_commit


# ---------------------------------------------------------------------------------------------
# worker-side: process init, per-sample render, timeout, parent-death handling
# ---------------------------------------------------------------------------------------------

class RenderTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise RenderTimeout("render exceeded its configured timeout")


def _set_pdeathsig():
    """Best-effort: ask the kernel to SIGTERM this worker if its parent dies (including a
    SIGKILLed parent), so a killed orchestrator never leaves rendering workers running. Linux
    only (uses prctl); silently does nothing elsewhere.

    Known narrow window: this only takes effect once THIS call has run, which is after the
    worker process's own (numpy/scipy/numba) import chain has finished bootstrapping. A worker
    killed together with its parent during that bootstrap -- before reaching this line -- has not
    armed PDEATHSIG yet and could survive as an orphan. Verified empirically (see
    test_process_hygiene_no_orphans_after_sigkill): once a worker is past init and into steady
    task execution, PDEATHSIG reliably terminates it within well under a second of the parent's
    death, by any means (SIGKILL included) -- the vulnerable window is bootstrap-only.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass


_WORKER = {}


def _worker_init(comp_dict, keep, timeout_s):
    _set_pdeathsig()
    assert "torch" not in sys.modules, "round-generation workers must never import torch"
    # scipy.fft's own thread count is a separate knob from OMP_NUM_THREADS et al (set in the
    # PARENT's environment before the pool was created, so spawned children inherit it before
    # numpy/BLAS ever loads) -- this one has to be set here, per-process, after import.
    render_mod.FFT_WORKERS = 1
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
    comp = CompositionConfig.from_dict(comp_dict)
    _WORKER["comp"] = comp
    _WORKER["struct"] = build_struct(comp)
    _WORKER["keep"] = keep
    _WORKER["timeout_s"] = timeout_s


def _seed_triple(master_seed, round_idx, index, attempt):
    """Three independent per-sample seeds (world-building nuisance randomness, theta sampling,
    tape/render), spawned from one SeedSequence([master_seed, round, index, attempt]) so output
    is a pure function of these four ints -- never of worker id or completion order."""
    ss = np.random.SeedSequence([int(master_seed), int(round_idx), int(index), int(attempt)])
    ss_world, ss_theta, ss_tape = ss.spawn(3)
    tape_seed = int(ss_tape.generate_state(1)[0])
    return ss_world, ss_theta, tape_seed


MOCK_RENDERER_ENV = "ROUNDS_MOCK_RENDERER"


def mock_renderer_enabled():
    return os.environ.get(MOCK_RENDERER_ENV) == "1"


def _mock_fail_indices():
    raw = os.environ.get("ROUNDS_MOCK_FAIL_INDICES", "")
    return {int(x) for x in raw.split(",") if x}


def _mock_sleep_s():
    return float(os.environ.get("ROUNDS_MOCK_SLEEP_S", "0") or 0)


def _mock_render_frame(comp, u, tape_seed):
    """Fast, deterministic stand-in for the real generator, used by the test suite so it isn't
    paying ~15s/frame. Small (16x16xC) frames whose channel c's mean level is linear in u[c]
    (cyclically, if there are more channels than theta dims) -- EXCEPT the last theta dimension,
    which is never read by any channel, so it has zero effect on the image. That deliberate
    no-effect parameter is what the end-to-end smoke test checks ends up marked saturated by the
    policy rather than chased forever. Selected via the ROUNDS_MOCK_RENDERER=1 env var (sur-
    vives `spawn` into worker processes, unlike a monkeypatched module attribute)."""
    h = w = 16
    rng = np.random.default_rng(tape_seed)
    n = len(u)
    levels = np.array([500.0 + 3000.0 * u[c % max(n - 1, 1)] for c in range(comp.n_markers)])
    img = rng.poisson(lam=np.broadcast_to(levels, (h, w, comp.n_markers))).astype(np.float64)
    img = np.clip(img, 0, 65535).astype(np.uint16)
    mask = np.zeros((h, w), dtype=np.int32)
    return img, mask


def _render_attempt(index, master_seed, round_idx, attempt, proposal, ps, keep):
    comp, struct, timeout_s = _WORKER["comp"], _WORKER["struct"], _WORKER["timeout_s"]
    ss_world, ss_theta, tape_seed = _seed_triple(master_seed, round_idx, index, attempt)
    theta_rng = np.random.default_rng(ss_theta)
    u, component_id = sample_theta(ps, proposal, theta_rng)
    if timeout_s and hasattr(signal, "SIGALRM"):
        signal.alarm(max(1, int(math.ceil(timeout_s))))
    try:
        if mock_renderer_enabled():
            if _mock_sleep_s():
                time.sleep(_mock_sleep_s())
            if index in _mock_fail_indices():
                raise ValueError(f"index {index}: forced failure (ROUNDS_MOCK_FAIL_INDICES)")
            img, mask = _mock_render_frame(comp, u, tape_seed)
        else:
            # generate_world draws real nuisance randomness (marker defaults, per-type baseline
            # expression, tissue fractions) as a side effect of construction -- exactly
            # mirroring gen_dataset.draw_one's own world_rng usage, this sample's own draw uses
            # its own independent seed, not the struct-seeded one ParamSpace's bounds came from.
            world = _build_world(comp, struct, np.random.default_rng(ss_world))
            ps.set_vector(world, u)
            img, mask, _geojson = render_frame(world, seed=tape_seed,
                                               with_artifacts=comp.with_artifacts)
    finally:
        if timeout_s and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    n_constant = int(sum(1 for c in range(img.shape[-1]) if img[..., c].std() == 0))
    if not np.isfinite(u).all() or not np.all((u >= 0.0) & (u <= 1.0)):
        raise ValueError(f"index {index}: theta left [0,1] or is non-finite")
    if img.dtype != np.uint16:
        raise ValueError(f"index {index}: image dtype {img.dtype}, expected uint16")
    return dict(u=u.astype(np.float32), img=img, mask=mask, component_id=component_id,
               n_constant=n_constant)


def _build_world(comp, struct, rng):
    from gen_dataset import generate_world
    return generate_world(struct, rng=rng, with_artifacts=comp.with_artifacts,
                          max_aggregates=comp.max_aggregates, max_fussels=comp.max_fussels,
                          max_detach_cover=comp.max_detach_cover)


def _render_one_with_retries(args):
    index, master_seed, round_idx, proposal_dict, max_retries, ps_dict = args
    proposal = Proposal.from_dict(proposal_dict)
    ps = ParamSpace.from_dict(ps_dict)
    keep = _WORKER["keep"]
    last_err = None
    for attempt in range(max_retries):
        try:
            out = _render_attempt(index, master_seed, round_idx, attempt, proposal, ps, keep)
            out["index"] = index
            out["attempts"] = attempt + 1
            return out
        except (ValueError, RenderTimeout) as e:
            last_err = f"{type(e).__name__}: {e}"
    return dict(index=index, error=last_err, attempts=max_retries)


# ---------------------------------------------------------------------------------------------
# parent-side: manifest, shards, progress/resumability, disk check
# ---------------------------------------------------------------------------------------------


def _frame_shape(comp):
    return (cfg.TISSUE_VOL[1], cfg.TISSUE_VOL[2], comp.n_markers)


def _check_disk_space(path, n, bytes_per_image, margin=1.25):
    need = int(n * bytes_per_image * margin)
    free = shutil.disk_usage(path).free
    if free < need:
        raise RuntimeError(
            f"not enough free disk space at {path}: need ~{need / 1e9:.2f}GB "
            f"({n} samples x {bytes_per_image} B/image x {margin} margin), "
            f"only {free / 1e9:.2f}GB free"
        )


def _progress_path(round_dir):
    return os.path.join(round_dir, "_progress.json")


def _load_progress(round_dir):
    p = _progress_path(round_dir)
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        d["shards_done"] = set(d["shards_done"])
        return d
    return dict(next_candidate_index=0, valid_count=0, shards_done=set())


def _save_progress(round_dir, progress):
    tmp = _progress_path(round_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(next_candidate_index=progress["next_candidate_index"],
                       valid_count=progress["valid_count"],
                       shards_done=sorted(progress["shards_done"])), f)
    os.replace(tmp, _progress_path(round_dir))


def _shard_paths(round_dir, shard_idx):
    stem = os.path.join(round_dir, f"shard_{shard_idx:04d}")
    return stem + ".images.npy", stem + ".npz"


def _clean_incomplete_shards(round_dir, n_shards, progress):
    """A shard is only trustworthy if BOTH its files exist AND progress.json says it's done
    (progress.json is only updated after the atomic rename below succeeds) -- anything else
    (leftover .tmp files, or a shard whose files exist but isn't marked done because the
    process died between writing the shard and saving progress) gets discarded and redone."""
    for shard_idx in range(n_shards):
        img_path, npz_path = _shard_paths(round_dir, shard_idx)
        for p in (img_path + ".tmp", npz_path + ".tmp"):
            if os.path.exists(p):
                os.remove(p)
        if shard_idx not in progress["shards_done"]:
            for p in (img_path, npz_path):
                if os.path.exists(p):
                    os.remove(p)


def _write_shard(round_dir, shard_idx, images, thetas, masks, comp_ids, n_constants):
    img_path, npz_path = _shard_paths(round_dir, shard_idx)
    img_arr = np.stack(images, axis=0).astype(np.uint16)
    tmp_img = img_path + ".tmp"
    # np.save auto-appends ".npy" to a bare string path that doesn't already end in ".npy",
    # which would silently mangle the ".tmp" suffix -- pass a file object to bypass that.
    with open(tmp_img, "wb") as f:
        np.save(f, img_arr)
    npz_kwargs = dict(u=np.stack(thetas, axis=0).astype(np.float32),
                      component_id=np.array(comp_ids, dtype=np.int32),
                      n_constant_channels=np.array(n_constants, dtype=np.int32))
    if masks is not None:
        npz_kwargs["mask"] = np.stack(masks, axis=0).astype(np.int32)
    tmp_npz = npz_path + ".tmp"
    with open(tmp_npz, "wb") as f:
        np.savez_compressed(f, **npz_kwargs)
    os.replace(tmp_img, img_path)
    os.replace(tmp_npz, npz_path)


def _log_failure(failures_path, index, error, comp):
    with open(failures_path, "a") as f:
        f.write(json.dumps(dict(index=index, error=error, n_markers=comp.n_markers,
                                n_types=comp.n_types, ts=time.time())) + "\n")


# ---------------------------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------------------------

def ensure_param_space(run_dir, comp, keep=None):
    """The ParamSpace for `run_dir` -- built fresh from `comp` and compared against whatever was
    persisted by an earlier round/eval-set in this run, or persisted now if this is the first
    call. Shared by generate_round and make_eval_set so both address theta identically."""
    os.makedirs(run_dir, exist_ok=True)
    ps = build_param_space(comp, keep=keep)
    ps_path = os.path.join(run_dir, "param_space.json")
    if os.path.exists(ps_path):
        existing = ParamSpace.load(ps_path)
        if existing.paths != ps.paths:
            raise RuntimeError(
                "ParamSpace layout mismatch with a previous round/eval-set in this run -- the "
                "composition config must not change within a run (Phase 1 assumes exactly one "
                "composition)"
            )
        return existing
    comp.save(os.path.join(run_dir, "composition.json"))
    ps.save(ps_path)
    return ps


def generate_samples(output_dir, master_seed, namespace_id, n, proposal, comp, gen_cfg, ps,
                     keep=None):
    """Core resumable, parallel sample generation, writing shards/manifest/failures/progress
    under `output_dir`. Shared by `generate_round` (namespace_id = round index) and
    `make_eval_set` (namespace_id = a fixed per-kind sentinel, disjoint from every real round
    index -- see eval_sets.py -- so an eval set's seeds can never collide with a training
    round's). Returns a dict of what happened; the caller adds its own framing fields (round vs.
    kind, git commit, etc.) and decides the manifest/meta filenames.
    """
    workers = int(gen_cfg.get("workers") or max(1, (os.cpu_count() or 2) - 2))
    shard_size = int(gen_cfg["shard_size"])
    render_timeout_s = float(gen_cfg["render_timeout_s"])
    max_extra_attempts_frac = float(gen_cfg["max_extra_attempts_frac"])
    store_masks = bool(gen_cfg.get("store_masks", False))
    max_retries = int(gen_cfg.get("max_retries", 5))

    os.makedirs(output_dir, exist_ok=True)
    bytes_per_image = int(np.prod(_frame_shape(comp))) * 2
    _check_disk_space(output_dir, n, bytes_per_image)

    n_shards = math.ceil(n / shard_size)
    progress = _load_progress(output_dir)
    _clean_incomplete_shards(output_dir, n_shards, progress)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest_is_new = not os.path.exists(manifest_path)
    failures_path = os.path.join(output_dir, "failures.jsonl")

    max_extra = int(math.ceil(n * max_extra_attempts_frac))
    extra_used = 0
    candidate_index = progress["next_candidate_index"]
    valid_count = progress["valid_count"]

    t0 = time.time()
    ctx = mp.get_context("spawn")
    ps_dict = ps.to_dict()
    with open(manifest_path, "a", newline="") as mf:
        writer = csv.writer(mf)
        if manifest_is_new:
            writer.writerow(["index", "shard", "position", "component_id", "attempts",
                             "n_constant_channels"])

        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init,
                                 initargs=(comp.to_dict(), keep, render_timeout_s)) as ex:
            for shard_idx in range(n_shards):
                if shard_idx in progress["shards_done"]:
                    continue
                target = min(shard_size, n - shard_idx * shard_size)
                images, thetas, masks, comp_ids, n_consts, rows = [], [], [], [], [], []

                while len(images) < target:
                    need = target - len(images)
                    batch = list(range(candidate_index, candidate_index + need))
                    candidate_index += need
                    tasks = [(i, master_seed, namespace_id, proposal.to_dict(), max_retries,
                             ps_dict) for i in batch]
                    futures = [ex.submit(_render_one_with_retries, t) for t in tasks]
                    for i, fut in zip(batch, futures):
                        result = fut.result()
                        if "error" in result:
                            _log_failure(failures_path, i, result["error"], comp)
                            extra_used += 1
                            if extra_used > max_extra:
                                raise RuntimeError(
                                    f"namespace {namespace_id}: exceeded "
                                    f"max_extra_attempts_frac={max_extra_attempts_frac} "
                                    f"(cap {max_extra} extra candidates) trying to reach n={n}; "
                                    f"only {valid_count + len(images)} valid samples produced. "
                                    f"See {failures_path}."
                                )
                            continue
                        images.append(result["img"])
                        thetas.append(result["u"])
                        comp_ids.append(result["component_id"])
                        n_consts.append(result["n_constant"])
                        if store_masks:
                            masks.append(result["mask"])
                        rows.append((i, shard_idx, len(images) - 1, result["component_id"],
                                    result["attempts"], result["n_constant"]))

                _write_shard(output_dir, shard_idx, images, thetas,
                            masks if store_masks else None, comp_ids, n_consts)
                for row in rows:
                    writer.writerow(row)
                mf.flush()
                valid_count += len(images)
                progress["shards_done"].add(shard_idx)
                progress["next_candidate_index"] = candidate_index
                progress["valid_count"] = valid_count
                _save_progress(output_dir, progress)

    elapsed = time.time() - t0
    return dict(
        n=n, valid_count=valid_count, n_shards=n_shards, candidate_indices_used=candidate_index,
        extra_attempts_used=extra_used, max_extra_attempts=max_extra, master_seed=master_seed,
        workers=workers, shard_size=shard_size, render_timeout_s=render_timeout_s,
        store_masks=store_masks, elapsed_s=elapsed, bytes_per_image=bytes_per_image,
        bytes_total=valid_count * bytes_per_image,
    )


def generate_round(run_dir, cfg_dict, round_idx, n, proposal, repo_dir=None):
    gen_cfg = cfg_dict["generation"]
    comp = CompositionConfig.from_dict(cfg_dict["phase1"]["composition"])
    master_seed = int(cfg_dict["run"]["master_seed"])
    keep = cfg_dict.get("keep")  # None -> composition.build_param_space uses cfg.KEEP

    ps = ensure_param_space(run_dir, comp, keep=keep)
    round_dir = os.path.join(run_dir, "rounds", f"round_{round_idx:03d}")
    result = generate_samples(round_dir, master_seed, round_idx, n, proposal, comp, gen_cfg, ps,
                              keep=keep)

    meta = dict(round=round_idx, **result,
               git_commit=git_commit(repo_dir or os.path.dirname(os.path.abspath(__file__))),
               config_hash=config_hash(cfg_dict), n_fitted_params=len(ps))
    proposal.save(os.path.join(round_dir, "proposal_used.json"))
    with open(os.path.join(round_dir, "round_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--proposal", default=None, help="path to a proposal_*.json; default: "
                    "prior-only")
    a = ap.parse_args()

    cfg_dict = load_config(a.config)
    n = a.n
    if n is None:
        phase1 = cfg_dict["phase1"]
        n = phase1["n0"] if a.round == 0 else phase1["n_per_round"]
    if a.proposal:
        proposal = Proposal.load(a.proposal)
    else:
        pcfg = cfg_dict.get("proposal", {})
        proposal = Proposal.prior_only(a.round, alpha_prior=pcfg.get("alpha_prior", 1.0),
                                       bins=pcfg.get("bins", 10), widen=pcfg.get("widen", False))

    meta = generate_round(a.run_dir, cfg_dict, a.round, n, proposal)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
