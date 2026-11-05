"""Model entry points. The inversion model is `model_v3.ModelV3` (plan Phase 4); the old flat-theta
architectures (ParamNet / SetParamNet) and their training/eval CLI have been removed -- v3 is the
only model. This module keeps just the input transform every consumer shares and a thin dispatch.
"""
import numpy as np

import config as cfg

ADU_MAX = cfg.ADU_MAX
LOGVAR_MIN, LOGVAR_MAX = cfg.LOGVAR_MIN, cfg.LOGVAR_MAX


def to_input(adu):
    """uint16 ADU -> network input: a single global log transform, no per-channel normalisation.
    Kept for reference / tooling; the v3 path uses `normalize_markers` instead (per-channel)."""
    x = np.asarray(adu, np.float32)
    return (np.log1p(np.clip(x, 0, ADU_MAX)) / np.float32(np.log1p(ADU_MAX))).astype(np.float32)


def normalize_markers(adu, pct=99.5):
    """Per-marker normalisation for the v3 model (plan upgrade).

    `adu`: [H, W, M] uint16. Each marker is scaled to [0, 1] by its own high percentile (robust to
    bright aggregate specks), so channels become COMPARABLE -- a 10-100x brighter DAPI no longer
    dominates the dynamic range. The absolute scale that per-channel normalisation removes is then
    handed back as a per-marker **expression strength** `log(mean_m / overall_mean)` (overall_mean =
    the mean of the per-marker means), so the expression-level target stays recoverable and the model
    can tell a strongly-expressed channel from a dim / background-only one. Returns
    (norm [H, W, M] float32, strength [M] float32)."""
    x = np.asarray(adu, np.float32)
    H, W, M = x.shape
    flat = x.reshape(-1, M)
    hi = np.percentile(flat, pct, axis=0) + 1e-6                     # per-marker robust max
    norm = np.clip(flat / hi, 0.0, 1.0).reshape(H, W, M).astype(np.float32)
    mean_m = flat.mean(0)                                            # per-marker mean intensity
    overall = float(mean_m.mean()) + 1e-6
    strength = np.log(mean_m / overall + 1e-3).astype(np.float32)    # relative brightness, log scale
    return norm, strength


def build_model(arch="v3", **kwargs):
    """Build the inversion model. Only `v3` exists now."""
    if arch == "v3":
        from model_v3 import ModelV3
        return ModelV3(**kwargs)
    raise ValueError(f"unknown model arch {arch!r} (only 'v3' exists)")
