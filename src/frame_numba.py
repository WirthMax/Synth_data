"""Fused Numba kernel for `body_frame`: image coords -> (rho, phi) in a cell's body frame.

`body_frame_numba(grid, rot, centre, stretch)` centres, rotates (`d @ rot`), unstretches, then
returns the radius `rho = ||d||` and the unit direction `phi = d/rho` in one nogil pass over the
flattened patch grid -- no (P,3) temporaries, all float32. This is the per-cell coordinate
transform, called on every patch in both build passes.

Import guard: if numba is unavailable, `body_frame_numba` is None and scene.py falls back to numpy.
"""
import numpy as np

try:
    from numba import njit, prange

    @njit(cache=True, parallel=True, fastmath=True, nogil=True)
    def _kernel(g, centre, rot, stretch):
        P = g.shape[0]
        rho = np.empty(P, np.float32)
        phi = np.empty((P, 3), np.float32)
        cx = centre[0]; cy = centre[1]; cz = centre[2]
        # d @ rot: out_k = sum_j d_j * rot[j, k]
        r00 = rot[0, 0]; r01 = rot[0, 1]; r02 = rot[0, 2]
        r10 = rot[1, 0]; r11 = rot[1, 1]; r12 = rot[1, 2]
        r20 = rot[2, 0]; r21 = rot[2, 1]; r22 = rot[2, 2]
        s0 = stretch[0]; s1 = stretch[1]; s2 = stretch[2]
        for i in prange(P):
            dx = g[i, 0] - cx; dy = g[i, 1] - cy; dz = g[i, 2] - cz
            rx = (dx * r00 + dy * r10 + dz * r20) / s0
            ry = (dx * r01 + dy * r11 + dz * r21) / s1
            rz = (dx * r02 + dy * r12 + dz * r22) / s2
            rr = np.sqrt(rx * rx + ry * ry + rz * rz)
            inv = 1.0 / max(rr, 1e-6)
            rho[i] = rr
            phi[i, 0] = rx * inv; phi[i, 1] = ry * inv; phi[i, 2] = rz * inv
        return rho, phi

    def body_frame_numba(grid, rot, centre, stretch):
        """grid (...,3) -> rho (...), phi (...,3). `rot` is the 3x3 body rotation (identity if none)."""
        shp = grid.shape[:-1]
        g = np.ascontiguousarray(grid.reshape(-1, 3), np.float32)
        rho, phi = _kernel(g, np.asarray(centre, np.float32),
                           np.ascontiguousarray(rot, np.float32),
                           np.asarray(stretch, np.float32))
        return rho.reshape(shp), phi.reshape(shp + (3,))

except ImportError:  # numba missing -> scene.py uses its numpy fallback
    body_frame_numba = None
