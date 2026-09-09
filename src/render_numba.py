"""Fused Numba kernel for the marker localisation envelope.

`localization_numba(x, mu, width, sharp, floor)` computes (1-floor)*exp(-|(x-mu)/width|**sharp)+floor
in a single pass -- no temporaries, no full-array np.power, and an early-out where the exponent
underflows (which also avoids the overflow that the plain numpy form triggers for large |x|).

Import guard: if numba is unavailable, `localization_numba` is None and render.py falls back to numpy.
"""
import numpy as np

try:
    from numba import njit, prange

    @njit(cache=True, parallel=True, fastmath=True, nogil=True)
    def _kernel(x, mu, inv_width, sharp, floor, one_minus_floor):
        n = x.shape[0]
        out = np.empty(n, np.float32)
        for i in prange(n):
            a = abs((x[i] - mu) * inv_width)
            e = a ** sharp
            # exp(-60) ~ 1e-26: below float32 resolution against the floor, so clamp and skip
            out[i] = floor if e > 60.0 else one_minus_floor * np.exp(-e) + floor
        return out

    def localization_numba(x, mu, width, sharp, floor):
        shp = x.shape
        flat = np.ascontiguousarray(x.reshape(-1), np.float32)
        out = _kernel(flat, np.float32(mu), np.float32(1.0 / width),
                      np.float32(sharp), np.float32(floor), np.float32(1.0 - floor))
        return out.reshape(shp)

except ImportError:  # numba missing -> render.py uses its numpy fallback
    localization_numba = None
