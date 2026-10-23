"""Model entry points. The inversion model is `model_v3.ModelV3` (plan Phase 4); the old flat-theta
architectures (ParamNet / SetParamNet) and their training/eval CLI have been removed -- v3 is the
only model. This module keeps just the input transform every consumer shares and a thin dispatch.
"""
import numpy as np

import config as cfg

ADU_MAX = cfg.ADU_MAX
LOGVAR_MIN, LOGVAR_MAX = cfg.LOGVAR_MIN, cfg.LOGVAR_MAX


def to_input(adu):
    """uint16 ADU -> network input: a single global log transform, no per-channel normalisation, so
    absolute brightness (which the expression targets depend on) survives. float32 explicitly -- a
    numpy float64 scalar would upcast and torch would then refuse it against float32 weights."""
    x = np.asarray(adu, np.float32)
    return (np.log1p(np.clip(x, 0, ADU_MAX)) / np.float32(np.log1p(ADU_MAX))).astype(np.float32)


def build_model(arch="v3", **kwargs):
    """Build the inversion model. Only `v3` exists now."""
    if arch == "v3":
        from model_v3 import ModelV3
        return ModelV3(**kwargs)
    raise ValueError(f"unknown model arch {arch!r} (only 'v3' exists)")
