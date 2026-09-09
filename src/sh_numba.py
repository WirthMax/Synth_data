"""Fused Numba kernel for the standard real SH basis (l_min=2, L=4).

`sh_eval_numba(u, c)` computes sum_j c_j Y_j(u) in a single pass per point -- no 21 temporary
arrays, no per-term dtype casts -- which is the hot path for the per-voxel boundary evaluations
and the batched nucleus-fit scans in scene.py. Mirrors `_sh_terms`/`sh_eval` exactly for the 21
coefficients of the (l=2..4) basis.

Import guard: if numba is unavailable, `sh_eval_numba` is None and scene.py falls back to numpy.
"""
import math
import numpy as np

# Leading normalisation constant of each of the 21 basis terms, in (l, m) order for l = 2..4.
_p = math.pi
_K = np.array([
    0.5 * math.sqrt(15 / _p),        # 0  l=2
    0.5 * math.sqrt(15 / _p),        # 1
    0.25 * math.sqrt(5 / _p),        # 2
    0.5 * math.sqrt(15 / _p),        # 3
    0.25 * math.sqrt(15 / _p),       # 4
    0.25 * math.sqrt(35 / (2 * _p)), # 5  l=3
    0.5 * math.sqrt(105 / _p),       # 6
    0.25 * math.sqrt(21 / (2 * _p)), # 7
    0.25 * math.sqrt(7 / _p),        # 8
    0.25 * math.sqrt(21 / (2 * _p)), # 9
    0.25 * math.sqrt(105 / _p),      # 10
    0.25 * math.sqrt(35 / (2 * _p)), # 11
    0.75 * math.sqrt(35 / _p),       # 12  l=4
    0.75 * math.sqrt(35 / (2 * _p)), # 13
    0.75 * math.sqrt(5 / _p),        # 14
    0.75 * math.sqrt(5 / (2 * _p)),  # 15
    (3 / 16) * math.sqrt(1 / _p),    # 16
    0.75 * math.sqrt(5 / (2 * _p)),  # 17
    (3 / 8) * math.sqrt(5 / _p),     # 18
    0.75 * math.sqrt(35 / (2 * _p)), # 19
    (3 / 16) * math.sqrt(35 / _p),   # 20
], np.float64)

try:
    from numba import njit, prange

    @njit(cache=True, parallel=True, fastmath=True)
    def _kernel(u, c):
        n = u.shape[0]
        out = np.empty(n, np.float32)
        for i in prange(n):
            x = u[i, 0]; y = u[i, 1]; z = u[i, 2]
            xx = x * x; yy = y * y; zz = z * z
            s  = c[0]  * _K[0]  * (x * y)
            s += c[1]  * _K[1]  * (y * z)
            s += c[2]  * _K[2]  * (3.0 * zz - 1.0)
            s += c[3]  * _K[3]  * (x * z)
            s += c[4]  * _K[4]  * (xx - yy)
            s += c[5]  * _K[5]  * (y * (3.0 * xx - yy))
            s += c[6]  * _K[6]  * (x * y * z)
            s += c[7]  * _K[7]  * (y * (5.0 * zz - 1.0))
            s += c[8]  * _K[8]  * (z * (5.0 * zz - 3.0))
            s += c[9]  * _K[9]  * (x * (5.0 * zz - 1.0))
            s += c[10] * _K[10] * (z * (xx - yy))
            s += c[11] * _K[11] * (x * (xx - 3.0 * yy))
            s += c[12] * _K[12] * (x * y * (xx - yy))
            s += c[13] * _K[13] * (y * z * (3.0 * xx - yy))
            s += c[14] * _K[14] * (x * y * (7.0 * zz - 1.0))
            s += c[15] * _K[15] * (y * z * (7.0 * zz - 3.0))
            s += c[16] * _K[16] * (35.0 * zz * zz - 30.0 * zz + 3.0)
            s += c[17] * _K[17] * (x * z * (7.0 * zz - 3.0))
            s += c[18] * _K[18] * ((xx - yy) * (7.0 * zz - 1.0))
            s += c[19] * _K[19] * (x * z * (xx - 3.0 * yy))
            s += c[20] * _K[20] * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy))
            out[i] = s
        return out

    def sh_eval_numba(u, c):
        """sum_j c_j Y_j(u) for the l=2..4 basis. `u` is (..., 3); returns (...) float32."""
        shp = u.shape[:-1]
        flat = np.ascontiguousarray(u.reshape(-1, 3), np.float32)
        return _kernel(flat, np.ascontiguousarray(c, np.float64)).reshape(shp)

except ImportError:  # numba missing -> scene.py uses its numpy fallback
    sh_eval_numba = None
