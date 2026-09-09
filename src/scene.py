from collections import namedtuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from parameter import CellContext
from render import render_marker
import tissue_direction as tdir
from sh_numba import sh_eval_numba, budget_worst_numba
from frame_numba import body_frame_numba

# To prevent rerunning things if they already ran once
_QUAD = None
_FIT_DIRS = None


def _quad():
    global _QUAD
    if _QUAD is None:
        _QUAD = sphere_quadrature()
    return _QUAD

def fit_directions(n_theta=24, n_phi=48):
    """Cached unit directions for the containment tests."""
    global _FIT_DIRS
    if _FIT_DIRS is None or _FIT_DIRS[1] != (n_theta, n_phi):
        _FIT_DIRS = (sphere_quadrature(n_theta, n_phi)[0].astype(np.float64),
                     (n_theta, n_phi))
    return _FIT_DIRS[0]

def _sh_terms(u, L=4, l_min=2):
    """Yield real orthonormal spherical harmonics one at a time, in (l, m) order.
    Assumes `u` is a normalized Cartesian vector array: u[..., 0]=x, u[..., 1]=y, u[..., 2]=z.
    """
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    xx, yy, zz = x * x, y * y, z * z
    p = np.pi

    if l_min <= 0 <= L:
        yield np.broadcast_to(np.float64(0.5 * np.sqrt(1 / p)), x.shape)
    if l_min <= 1 <= L:
        c = np.sqrt(3 / (4 * p))
        yield c * y; yield c * z; yield c * x
    if l_min <= 2 <= L:
        yield 0.5 * np.sqrt(15 / p) * x * y
        yield 0.5 * np.sqrt(15 / p) * y * z
        yield 0.25 * np.sqrt(5 / p) * (3 * zz - 1.0)
        yield 0.5 * np.sqrt(15 / p) * x * z
        yield 0.25 * np.sqrt(15 / p) * (xx - yy)
    if l_min <= 3 <= L:
        yield 0.25 * np.sqrt(35 / (2 * p)) * y * (3 * xx - yy)
        yield 0.5 * np.sqrt(105 / p) * x * y * z
        yield 0.25 * np.sqrt(21 / (2 * p)) * y * (5 * zz - 1.0)
        yield 0.25 * np.sqrt(7 / p) * z * (5 * zz - 3.0)
        yield 0.25 * np.sqrt(21 / (2 * p)) * x * (5 * zz - 1.0)
        yield 0.25 * np.sqrt(105 / p) * z * (xx - yy)
        yield 0.25 * np.sqrt(35 / (2 * p)) * x * (xx - 3 * yy)
    if l_min <= 4 <= L:
        yield 0.75 * np.sqrt(35 / p) * x * y * (xx - yy)
        yield 0.75 * np.sqrt(35 / (2 * p)) * y * z * (3 * xx - yy)
        yield 0.75 * np.sqrt(5 / p) * x * y * (7 * zz - 1.0)
        yield 0.75 * np.sqrt(5 / (2 * p)) * y * z * (7 * zz - 3.0)
        yield (3 / 16) * np.sqrt(1 / p) * (35 * zz * zz - 30 * zz + 3.0)
        yield 0.75 * np.sqrt(5 / (2 * p)) * x * z * (7 * zz - 3.0)
        yield (3 / 8) * np.sqrt(5 / p) * (xx - yy) * (7 * zz - 1.0)
        yield 0.75 * np.sqrt(35 / (2 * p)) * x * z * (xx - 3 * yy)
        yield (3 / 16) * np.sqrt(35 / p) * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

def sh_eval(u, c, L=4, l_min=2, dtype=np.float32):
    """sum_j c_j Y_j(u), accumulated term by term so nothing large is materialised."""
    out = np.zeros(u.shape[:-1], dtype)
    sh_terms = list(_sh_terms(u, L, l_min))
    for cj, term in zip(c, sh_terms):
        if cj != 0.0:
            out += (cj * term).astype(dtype, copy=False)
    return out

def sh_basis(u, L=4, l_min=2):
    """Stacked basis, (..., n_lm). Only for small point sets -- see `_sh_terms`."""
    return np.stack(list(_sh_terms(u, L, l_min)), -1)

def sh_degrees(L=4, l_min=2):
    """The degrees of each basis column, so the spectrum can be applied per column."""
    return np.concatenate([np.full(2 * l + 1, l) for l in range(l_min, L + 1)])


def n_sh(L=4, l_min=2):
    """Compute the total number of spherical harmonics for given range of l."""
    return int(sh_degrees(L, l_min).size)


def sphere_quadrature(n_theta=48, n_phi=96):
    """Gauss-Legendre in cos(theta) x uniform in phi. Returns (u [N,3], w [N]), sum(w) = 4 pi."""
    ct, wct = np.polynomial.legendre.leggauss(n_theta)
    st = np.sqrt(np.maximum(1.0 - ct ** 2, 0.0))
    ph = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    u = np.stack([np.outer(st, np.cos(ph)),
                  np.outer(st, np.sin(ph)),
                  np.repeat(ct[:, None], n_phi, 1)], -1).reshape(-1, 3)
    w = np.repeat(wct[:, None], n_phi, 1).ravel() * (2.0 * np.pi / n_phi)
    return u, w

# finite-difference step shared by _rim_scale and its cached basis
_RIM_EPS = 1e-4


def _sh_eval(u, c, L=4, l_min=2):
    """General SH eval: fused numba kernel for the standard l=2..4 basis, numpy fallback else."""
    if sh_eval_numba is not None and (L, l_min) == (4, 2):
        return sh_eval_numba(u, c)
    return sh_eval(u, c, L, l_min)


class Boundary:
    """Support function r(u) = scale * exp(sum_j c_j Y_j(u)) of one contour.
    """
    __slots__ = ("c", "scale", "L", "l_min")

    def __init__(self, c, scale, L=4, l_min=2):
        self.c = np.ascontiguousarray(c, np.float64)
        self.scale = float(scale)
        self.L, self.l_min = L, l_min

    def __call__(self, u):
        return self.scale * np.exp(_sh_eval(u, self.c, self.L, self.l_min))

    def on_basis(self, B):
        return self.scale * np.exp(B @ self.c)


# Basis matrices on the fixed containment/quadrature directions are identical for every cell, so
# build each once and reuse. Keyed by the direction array's identity (the dir sets are themselves
# cached module globals, so their ids are stable).
_BASIS_BY_ID = {}


def _basis_for(dirs, L, l_min):
    """(B, B_pert) for a cached direction set: B on `dirs`, B_pert on its +/-eps perturbations."""
    key = (id(dirs), L, l_min)
    hit = _BASIS_BY_ID.get(key)
    if hit is None:
        e = np.eye(3) * _RIM_EPS
        p = np.concatenate([dirs[None] + e[:, None, :], dirs[None] - e[:, None, :]])   # (6, n, 3)
        p /= np.linalg.norm(p, axis=-1, keepdims=True)
        hit = _BASIS_BY_ID[key] = (sh_basis(dirs, L, l_min), sh_basis(p, L, l_min))
    return hit


def make_boundary(coeff, R, kappa, beta, L = 4, l_min = 2):
    """Return r(phi): the support function of one contour, volume normalised to (4/3)pi R^3.
Coeff holds all the randomness — one frozen row per cell.
In 2D the boundary is a Fourier series in the angle φ; in 3D it's the same idea with spherical harmonics over directions on the sphere.
The coefficients are amplitudes of global patterns, not points: each one perturbs the radius everywhere at once, at its own angular frequency.
w is a spectral envelope. beta tilts it (how much coarse vs fine detail), and it's then
renormalised so the total is 4πκ² — which makes rough mean exactly "SD of log-radius", independent of beta and L.
Multiplying coeff * w realises one specific cell's amplitudes: to be always positive, defined continuously in every direction.
The only thing left is the size. A quadrature over the sphere measures this shape's volume at scale = 1,
and scale is then set so the volume equals (4/3)πR³. So radius is the equivalent-sphere radius, whatever the roughness.
    """
    # compute the weight for each degree, such that all weights of a degree are equal
    w = sh_degrees(L, l_min).astype(float) ** -float(beta)
    # renormalization factor to ensure the area is correct
    # With this, changing beta only changes the kind of roughness, not the overall size of the cell.
    w = w * (kappa * np.sqrt(4.0 * np.pi) / np.sqrt((w ** 2).sum() + 1e-30))
    c = w * coeff

    # Compute quadrature to pinpoint size and then scale accordingly
    uq, wq = _quad()
    B_quad, _ = _basis_for(uq, L, l_min)
    gq = B_quad @ c
    # volume at scale = 1
    vol = float((np.exp(3.0 * gq) * wq).sum() / 3.0)
    scale = R * ((4.0 * np.pi / 3.0) / max(vol, 1e-30)) ** (1.0 / 3.0)
    return Boundary(c, scale, L, l_min)

def rotation_matrix(polar_deg=0.0, azim_deg=0.0, roll_deg=0.0):
    return R.from_euler('ZYZ', [azim_deg, polar_deg, roll_deg], degrees=True).as_matrix()


def _stretch(elong):
    """Volume-preserving ellipsoid semi-axes: `elong` along body-z, elong^-1/2 across.
    The product A * B^2 = 1 always, so elongation cannot leak into cell size.
    """
    A = float(elong)
    B = A ** -0.5
    return np.array([B, B, A])

def body_frame(grid, rot, centre = (0.0, 0.0, 0.0), elong=1.0):
    """Image coordinates -> (rho, phi) in the cell's rotated, un-stretched frame.
    Do this to prevent having to expensively re-derive the cells shape parameters """
    stretch = _stretch(elong)
    if body_frame_numba is not None:
        rot = np.eye(3) if rot is None else rot
        return body_frame_numba(grid, rot, centre, stretch)
    # numpy fallback: 1 centre, 2 rotate so the long axis is body-z, 3 unstretch ellipsoid
    d = (grid - np.asarray(centre, np.float32)).astype(np.float32, copy=False)
    if rot is not None:
        d = d @ np.asarray(rot, np.float32)
    d = d / stretch.astype(np.float32)
    rho = np.sqrt((d * d).sum(-1))
    return rho, d / np.maximum(rho, 1e-6)[..., None]
    


def centred_grid_3d(shape, spacing=1.0):
    """Full grid with the origin at the image centre -- the sandbox case in 3d."""
    nz, ny, nx = shape
    # set up bounding box of size nz, ny, nx but centered at 0,0,0. 
    # The spacing is the distance between pixels in each dimension.
    # Only relevant if the sampling is not isotropic, i.e. distances between 
    # pixels in z direction are larger.
    sz, sy, sx = (spacing,) * 3 if np.isscalar(spacing) else spacing
    z = (np.arange(nz, dtype=np.float32) - (nz - 1) / 2.0) * sz
    y = (np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0) * sy
    x = (np.arange(nx, dtype=np.float32) - (nx - 1) / 2.0) * sx
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    return np.stack([X, Y, Z], -1)


def patch_grid(centre_xyz, reach, shape):
    """Box of half-width `reach` around a centre, clipped to the volume."""
    nz, ny, nx = shape
    cx, cy, cz = centre_xyz
    return (slice(max(0, int(cz) - reach), min(nz, int(cz) + reach + 1)),
            slice(max(0, int(cy) - reach), min(ny, int(cy) + reach + 1)),
            slice(max(0, int(cx) - reach), min(nx, int(cx) + reach + 1)))


def reach_px(R, rough, elong = 1.0, grow=1.0, pad=2):
    """Furthest pixel this cell can claim, in image vox"""
    return int(np.ceil(R * grow * max(elong, 1.0 / elong) * np.exp(3.0 * rough))) + pad


def mix_harmonics(sh, sh2, corr):
    """Nuclear coefficients correlated with the cell's at level `corr`, without resampling."""
    c = float(np.clip(corr, 0.0, 1.0))
    s = np.sqrt(1.0 - c * c)
    return c * np.asarray(sh, float) + s * np.asarray(sh2, float)

def to_world(offset_body, rot=None, elong=1.0):
    """Body-frame (un-stretched) offset -> world offset: re-apply the stretch, then rotate.

    A body-frame length means the same thing at every orientation only if it is built here
    and mapped out, never added directly in world coordinates.
    """
    o = np.asarray(offset_body, float) * _stretch(elong)
    return o if rot is None else np.asarray(rot, float) @ o

def containment_residual(r_cell_fn, r_nuc_fn, off_hat, delta, rim=1.5, dirs=None,
                         euclid_rim=True, rim_eff=None, base=None):
    dirs = fit_directions() if dirs is None else dirs
    if base is None:
        B, _ = _basis_for(dirs, r_cell_fn.L, r_cell_fn.l_min)
        base = r_nuc_fn.on_basis(B)[:, None] * dirs
    if rim_eff is None:
        rim_eff = rim * _rim_scale(r_cell_fn, dirs) if euclid_rim else rim
    p = float(delta) * np.asarray(off_hat, float) + base
    rho = np.linalg.norm(p, axis=-1)
    return rho - r_cell_fn(p / np.maximum(rho, 1e-30)[..., None]) + rim_eff

def _rim_scale(r_cell_fn, dirs):
    _, B_pert = _basis_for(dirs, r_cell_fn.L, r_cell_fn.l_min)
    lg = np.log(np.maximum(r_cell_fn.on_basis(B_pert), 1e-30))                     # (6, n)
    d = (lg[:3] - lg[3:]) / (2 * _RIM_EPS)
    return np.sqrt(1.0 + (d * d).sum(axis=0))

def nuc_offset_budget(r_cell_fn, r_nuc_fn, off_hat, rim=1.5,
                      n_scan=33, n_fine=129, euclid_rim=True):
    """Largest offset delta (along off_hat) at which the nucleus still fits inside the cell with
    the rim clearance. The worst-direction residual is monotone increasing in delta (pushing the
    nucleus out can only worsen containment), so one batched coarse scan brackets the zero
    crossing and one batched fine scan locates it.
    """
    dirs = fit_directions(n_theta=24, n_phi=48)
    B, _ = _basis_for(dirs, r_cell_fn.L, r_cell_fn.l_min)
    off_hat = np.asarray(off_hat, float)
    base = r_nuc_fn.on_basis(B)[:, None] * dirs                                    # (n, 3)
    rim_eff = rim * _rim_scale(r_cell_fn, dirs) if euclid_rim else rim
    rc = r_cell_fn.on_basis(B)

    def worst(deltas):
        """Worst-direction residual for each offset in `deltas` (M,) -> (M,), one batched eval."""
        if budget_worst_numba is not None:
            re = rim_eff if np.ndim(rim_eff) else np.full(base.shape[0], float(rim_eff))
            # off_hat may be scalar (the sandbox tape's 2-D u_offdir): numpy broadcasts it across
            # all three axes, so expand to (3,) to match -- [s,s,s] == delta*s added per coord.
            off3 = np.broadcast_to(np.asarray(off_hat, np.float64), (3,))
            return budget_worst_numba(np.ascontiguousarray(deltas, np.float64),
                                      np.ascontiguousarray(off3, np.float64),
                                      np.ascontiguousarray(base, np.float64),
                                      np.ascontiguousarray(re, np.float64),
                                      r_cell_fn.c, float(r_cell_fn.scale))
        p = deltas[:, None, None] * off_hat + base                                # (M, n, 3)
        rho = np.linalg.norm(p, axis=-1)
        rcp = r_cell_fn(p / np.maximum(rho, 1e-30)[..., None])                     # (M, n)
        return (rho - rcp + rim_eff).max(axis=1)

    hi = max(float(np.max(rc)) - float(np.min(np.atleast_1d(rim_eff))), 0.0)
    if hi <= 0.0:
        return 0.0
    grid = np.linspace(0.0, hi, n_scan)
    over = np.flatnonzero(worst(grid) > 0.0)
    if over.size == 0:
        return float(hi)                      # fits at any offset (near-spherical cell)
    k = int(over[0])
    if k == 0:
        return 0.0                            # s0 < 1: infeasible even centred
    fine = np.linspace(grid[k - 1], grid[k], n_fine)
    fover = np.flatnonzero(worst(fine) > 0.0)
    return float(fine[fover[0] - 1]) if fover.size and fover[0] > 0 else float(fine[0])


def nuc_fit_scale(r_cell_fn, r_nuc_fn, rim=1.5, dirs=None, euclid_rim=True):
    """s0 = min_w (r_c(w) - rim_eff(w)) / r_n(w).

    s0 >= 1 means the CENTRED nucleus already fits with the full rim. s0 < 1 means `rough`,
    `nuc_frac` and `rim` are jointly infeasible for this cell and the clip in `cell_fields_3d`
    will bite at EVERY offset, including 0 -- no choice of `nuc_offset` can rescue it.

    Measured at radius = 24, rim = 1.5: at rough = 0.25 this affects 32% of cells for
    nuc_frac = 0.45, but 0% for nuc_frac <= 0.30.
    """
    dirs = fit_directions() if dirs is None else dirs
    B, _ = _basis_for(dirs, r_cell_fn.L, r_cell_fn.l_min)
    rim_eff = rim * _rim_scale(r_cell_fn, dirs) if euclid_rim else rim
    return float(np.min((r_cell_fn.on_basis(B) - rim_eff) / np.maximum(r_nuc_fn.on_basis(B), 1e-30)))


def cell_fields(rho_c, phi_c, r_c, rho_n=None, phi_n=None, r_n=None, rim=1.5, grow=1.0):
    """Masks and coordinates from two supports about POSSIBLY DIFFERENT centres.

    psi = rho - r(phi) is signed (negative inside), measured relative to each contour --
    which is what lets the two be compared even though they have different origins.
    The containment clip is applied to psi_n ONCE and both outputs derive from it, so the
    labelled nuclear edge and the tau = 0 level set stay the same curve.

    grow scales the CELL contour only (r_c -> grow * r_c), leaving the nucleus
    untouched. 1.0 = the free shape (the mask uses this). grow > 1 gives the packed body: d,
    cell and tau then reference the grown membrane, so a marker rendered on these fields fills
    the tessellated territory instead of just the free shape.
    """
    r_c = r_c * grow
    d = rho_c / np.maximum(r_c, 1e-6)
    out = {"d": d, "cell": d <= 1.0, "phi": phi_c, "rho": rho_c}
    if r_n is None:
        return out
    psi_c = rho_c - r_c
    psi_n = np.maximum(rho_n - r_n, psi_c + rim)
    out["nuc"] = psi_n <= 0.0
    out["tau"] = np.where(psi_n < 0.0,
                          rho_n / np.maximum(r_n, 1e-6) - 1.0,
                          psi_n / np.maximum(psi_n - psi_c, 1e-6))
    return out


# The grow-independent per-cell fit: boundaries, the offset direction, the nucleus centre and its
# room budget. Identical for the packing pass (grow=1) and the render pass (grow=GROW), so it is
# computed once and reused -- only the final cell_fields depends on grow.
Fit = namedtuple("Fit", "r_cell_fn r_nuc_fn off ncentre free s0 delta")


def _cell_fit(c_cell, c_nuc, L, l_min, nuc_corr, nuc_frac, radius, rough, beta, rot, elong,
              rim, nuc_offset, off_dir, off_mag, centre, rough_nuc, beta_nuc, euclid_rim,
              nuc_fit_floor):
    """The expensive grow-independent work: build both boundaries, fit the nucleus offset."""
    cn = mix_harmonics(c_cell, c_nuc, nuc_corr)
    r_cell_fn = make_boundary(coeff = c_cell, R = radius, kappa = rough,
                              beta = beta, L = L, l_min = l_min)
    if rough_nuc is None:
        rough_nuc = rough * 0.6
    if rough_nuc is None:
        beta_nuc = beta
    r_nuc_fn = make_boundary(coeff = cn, R = radius * nuc_frac, kappa = rough_nuc,
                             beta = beta_nuc, L=L, l_min = l_min)
    off = np.asarray(off_dir, float)
    off = off / (np.linalg.norm(off) + 1e-30)

    s0 = nuc_fit_scale(r_cell_fn, r_nuc_fn, rim, euclid_rim=euclid_rim)
    if nuc_fit_floor is not None and s0 < 1.0:
        # Opt-in rescue: shrink the nucleus uniformly. Smooth and isotropic, whereas the clip in
        # `cell_fields_3d` would instead cut a flat facet. Off by default because silently
        # shrinking nuclei would break what `nuc_frac` means.
        # A uniform scale k factors straight through: k * scale * exp(...) is just a Boundary with
        # scale*k, so the on_basis fast path keeps working.
        _k = max(float(s0), float(nuc_fit_floor))
        r_nuc_fn = Boundary(r_nuc_fn.c, r_nuc_fn.scale * _k, r_nuc_fn.L, r_nuc_fn.l_min)

    # Compute freespace to membrane with rim space
    free = nuc_offset_budget(r_cell_fn, r_nuc_fn, off, rim=rim,
                        n_scan=33, n_fine=129, euclid_rim=True)
    delta = nuc_offset * free * off_mag
    ncentre = np.asarray(centre, float) + to_world(off * delta, rot, elong)
    return Fit(r_cell_fn, r_nuc_fn, off, ncentre, float(free), float(s0), float(delta))


def _cell_and_nucleus(c_cell, c_nuc, grid, L, l_min, nuc_corr, nuc_frac, radius,
                      rough, beta, rot, elong, rim, nuc_offset, off_dir, off_mag, grow,
                      centre = (0.0, 0.0, 0.0), rough_nuc = None, beta_nuc=None,
                      euclid_rim=True, nuc_fit_floor=None, nuc_elong=None, fit=None):
    """Shared core: both wrappers do exactly this, only the grid and grow differ.

    body_grow scales the cell contour (not the nucleus); see cell_fields. Default 1.0 = the
    free shape used everywhere for the mask; > 1 gives the packed body for marker rendering.

    `fit` is the grow-independent bundle from `_cell_fit`. Pass the one stored by the packing
    pass to skip re-fitting the nucleus (make_boundary/nuc_fit_scale/nuc_offset_budget) on the
    render pass -- the boundaries and centre are identical, only cell_fields(grow) changes.
    """
    if nuc_elong is None:
        nuc_elong = elong
    reused = fit is not None
    if not reused:
        fit = _cell_fit(c_cell, c_nuc, L, l_min, nuc_corr, nuc_frac, radius, rough, beta, rot,
                        elong, rim, nuc_offset, off_dir, off_mag, centre, rough_nuc, beta_nuc,
                        euclid_rim, nuc_fit_floor)
    r_cell_fn, r_nuc_fn = fit.r_cell_fn, fit.r_nuc_fn

    rho_c, phi_c = body_frame(grid = grid, rot = rot, centre = centre, elong=elong)
    rho_n, phi_n = body_frame(grid, rot = rot, centre = fit.ncentre, elong=nuc_elong)

    f = cell_fields(rho_c, phi_c, r_cell_fn(phi_c),
                    rho_n, phi_n, r_nuc_fn(phi_n),
                    rim, grow=grow)
    f["nuc_centre"] = fit.ncentre
    f["r_cell_fn"], f["r_nuc_fn"] = r_cell_fn, r_nuc_fn
    f["nuc_room"] = fit.free
    f["nuc_fit_scale"] = fit.s0
    f["fit"] = fit
    if not reused:
        # diagnostics only (nuc_shaved_*); the render pass reuses the fit and never reads them
        h = containment_residual(r_cell_fn, r_nuc_fn, fit.off, fit.delta, rim,
                                 dirs=_quad()[0], euclid_rim=euclid_rim)
        f["nuc_shaved_frac"] = float((h > 0.0).mean())
        f["nuc_shaved_max"] = float(h.max())
    return f


def generate_single_cell_3d(Tape, geom,
                            size=(128, 128, 128), spacing = 1.0, L=4, l_min=2, i = 0,
                            grow = 1.0, euclid_rim=True, nuc_fit_floor=None):
    """Sandbox: candidate i from the tape, centred in its own image."""
    grid = centred_grid_3d(size, spacing=spacing)
    rot = rotation_matrix(polar_deg=geom.POLAR_DEG.v, azim_deg=geom.AZIM_DEG.v,
                          roll_deg=geom.ROLL_DEG.v)

    return _cell_and_nucleus(
        c_cell = Tape["sh"][i],
        c_nuc = Tape["sh2"][i],
        grid = grid,
        centre = (0.0, 0.0, 0.0),
        rot = rot,
        radius = geom.RADIUS.v,
        nuc_frac = geom.NUC_FRAC.v,
        rough = geom.ROUGH.v,
        beta = geom.BETA.v,
        L = L, l_min = l_min,
        elong = geom.ELONG.v,
        nuc_elong = geom.NUC_ELONG.v,
        rim = geom.RIM.v,
        nuc_corr = geom.NUC_CORR.v,
        nuc_offset = geom.NUC_OFFSET.v,
        off_dir=Tape["u_offdir"][i], off_mag=Tape["u_offmag"][i],
        grow=grow,
        euclid_rim=euclid_rim,
        nuc_fit_floor=nuc_fit_floor,
        rough_nuc = geom.NUC_ROUGH.v,
        beta_nuc=geom.NUC_BETA.v
        )
        

### Tissue
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi
import dataclasses


def stamp_cell(tape, i, geom, grid, sl, centre_world, L = 4, l_min = 2,
               grow=1.0, fit=None):
    """Tissue: candidate i stamped at (cy, cx, cz) on a local patch.

    `grow` sizes the cell contour only; `body_grow` scales the rendered cell contour (see
    cell_fields). They are set together (body_grow=grow) to render markers on the packed body.
    `fit` reuses the packing pass's nucleus fit (see `_cell_and_nucleus`).
    """
    return _cell_and_nucleus(
        c_cell=tape["sh"][i], c_nuc=tape["sh2"][i], grid=grid[sl], L=L, l_min=l_min,
        nuc_corr=geom.NUC_CORR.v, nuc_frac=geom.NUC_FRAC.v, radius=geom.RADIUS.v,
        rough=geom.ROUGH.v, beta=geom.BETA.v,
        rot=rotation_matrix(geom.POLAR_DEG.v, geom.AZIM_DEG.v, geom.ROLL_DEG.v),
        elong=geom.ELONG.v, nuc_elong=geom.NUC_ELONG.v, rim=geom.RIM.v,
        nuc_offset=geom.NUC_OFFSET.v,
        off_dir=tape["u_offdir"][i], off_mag=tape["u_offmag"][i],
        grow=grow, centre=centre_world,
        rough_nuc=geom.NUC_ROUGH.v, beta_nuc=geom.NUC_BETA.v, fit=fit)


def vox_to_world(pts_xyz, shape, spacing):
    """(x,y,z) voxel indices -> the world coordinates centred_grid_3d uses."""
    nz, ny, nx = shape
    sz, sy, sx = spacing
    return (np.asarray(pts_xyz, float)
            - np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2])) * np.array([sx, sy, sz])

def cell_geometry(base, ttape, i, tg, n_dir=None, S=0.0, align=0.0):
    """This cell's own geometry: the type's shape plus its frozen size / orientation draws.
    """
    r = float(np.clip(base.RADIUS.v * np.exp(tg.SIZE_SIGMA.v * ttape["z_size"][i]),
                      base.RADIUS.lo, base.RADIUS.hi))
    nf = float(np.clip(base.NUC_FRAC.v * np.exp(tg.NUC_FRAC_SIGMA.v * ttape["z_nucfrac"][i]),
                       base.NUC_FRAC.lo, base.NUC_FRAC.hi))
    o = ttape["u_orient3"][i]
    
    polar, azim = 180.0 * o[0], 360.0 * o[1]
    a = float(align) * float(S)
    if n_dir is not None and a > 0.0:
        p, z = np.deg2rad(polar), np.deg2rad(azim)
        d = np.array([np.sin(p) * np.cos(z), np.sin(p) * np.sin(z), np.cos(p)])
        d = tdir.slerp_axis(d, n_dir, a)
        polar = float(np.rad2deg(np.arccos(np.clip(d[2], -1.0, 1.0))))
        azim = float(np.rad2deg(np.arctan2(d[1], d[0])) % 360.0)
    return dataclasses.replace(base, RADIUS=r, NUC_FRAC=nf, POLAR_DEG=float(polar),
                               AZIM_DEG=float(azim), ROLL_DEG=float(360.0 * o[2]))

def thin(pts, order, min_dist):
    keep = np.ones(len(pts), bool)
    for i, j in cKDTree(pts).query_pairs(float(min_dist), output_type="ndarray"):
        keep[i if order[i] > order[j] else j] = False
    return np.flatnonzero(keep)

def thin_density(pts, order, radii):
    """Matern type-II hard-core thinning with a per-candidateexclusion radius."""
    radii = np.asarray(radii, float)
    keep = np.ones(len(pts), bool)
    pairs = cKDTree(pts).query_pairs(float(radii.max()) if len(radii) else 0.0,
                                      output_type="ndarray")
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        conflict = np.linalg.norm(pts[i] - pts[j], axis=1) <= 0.5 * (radii[i] + radii[j])
        i, j = i[conflict], j[conflict]
        keep[np.where(order[i] > order[j], i, j)] = False
    return np.flatnonzero(keep)


def _smooth_z(w, XY_scale_vox=40.0, Z_scale_vox=40.0):
    """Smooth the frozen field and z-score it. The primitive psi and support_mask share."""
    z = gaussian_filter(w.astype(np.float32),
                        (float(Z_scale_vox), float(XY_scale_vox), float(XY_scale_vox)), truncate=3.0)
    return (z - z.mean()) / (z.std() + 1e-12)

def support_mask(w, XY_scale_vox=40.0, Z_scale_vox=40.0, cover=0.75, psi = None):
    """Smooth the frozen field, then threshold at a quantile so `cover` IS the realised fraction."""
    z = _smooth_z(w, XY_scale_vox, Z_scale_vox) if psi is None else np.asarray(psi, np.float32)
    return z >= np.quantile(z, 1.0 - float(np.clip(cover, 0.0, 1.0)))


def assign_types(info, tape, CellTypes, fractions, rule=None):
    """Decide each cell's type: the global Fractions, tilted by the structure Profile it sits in
    and by its cell type's Affinity for each named tissue feature."""
    type_names = list(CellTypes)
    f = np.asarray(fractions, float)
    flat = np.cumsum(f / np.sum(f))
    field = info.get("field")
    fmap = info.get("feat")
    lw = info.get("log_weights")
    refs = info.get("feat_refs", {})
    feat_names = list((field or {}).get("features", {})) if field is not None else []
    types = {}
    for n in info["labels_present"]:
        i = info["cand_idx"][n]
        feat = {}
        if field is None or fmap is None:
            cuts = flat
        else:
            feat = fmap[n]
            tilt = feat["m"] @ lw if lw is not None and len(lw) else 0.0
            aff = np.array([
                sum(CellTypes[t].affinity(nm) * (float(feat[nm]) - refs.get(nm, 0.0))
                    for nm in feat_names)
                for t in type_names])
            q = f * np.exp(tilt + aff)
            cuts = np.cumsum(q / q.sum())
        base = type_names[int(np.searchsorted(cuts, tape["u_type"][i]))]
        ctx = CellContext(label=n, index=i, centre=info["centres_vox"][n],
                          neigh_labels=info["neighbours"][n], types=types, feat=feat)
        types[int(n)] = base if rule is None else rule(ctx, base)
    return types

class TapeDict(dict):
    """dict that also allows attribute access, so it works wherever a Tape does."""
    __getattr__ = dict.__getitem__

def build_tissue(tape, TG, shape, base_geom, spacing, Panel, CellTypes, Fractions,
                 um_per_vox, L=4, l_min=2, rule=None, pool_bank=None):
    """Tissue in three phases: geometry, then type assignment, then appearance."""
    sup = support_mask(w = tape["support3"], XY_scale_vox = TG.SUPPORT_SCALE.v, 
                    Z_scale_vox = TG.SUPPORT_SCALE_Z.v,
                    cover = TG.COVER.v)
    keep = thin(tape["xyz"], tape["order"], TG.MIN_DIST.v)
    cx, cy, cz = tape["xyz"][keep].T
    nz, ny, nx = shape
    keep = keep[sup[np.clip(cz.astype(int), 0, nz - 1),
                    np.clip(cy.astype(int), 0, ny - 1),
                    np.clip(cx.astype(int), 0, nx - 1)]]
    
    grid = centred_grid_3d(shape, spacing)
    best = np.full(shape, np.inf, np.float32)
    labels = np.zeros(shape, np.int32)
    nuc_labels = np.zeros(shape, np.int32)
    tau_img = np.zeros(shape, np.float32)
    geoms, slices, centres_w, fits = {}, {}, {}, {}

    for n, i in enumerate(keep, start=1):
        g = cell_geometry(base_geom, tape, i, TG)
        sl = patch_grid(tape["xyz"][i],
                         reach_px(R=g.RADIUS.v, rough=g.ROUGH.v, elong=g.ELONG.v, grow=TG.GROW.v), 
                         shape)
        if any(s.stop - s.start <= 0 for s in sl):
            continue
        cw = vox_to_world(tape["xyz"][i], shape, spacing)
        f = stamp_cell(tape, i, g, grid, sl, cw, L = L, l_min = l_min, grow=1.0)

        # tesselation to decide which cell is closest to each pixel 
        win = (f["d"] < best[sl]) & (f["d"] <= TG.GROW.v) & sup[sl]
        best[sl] = np.where(win, f["d"], best[sl])
        labels[sl] = np.where(win, n, labels[sl])
        tau_img[sl] = np.where(win, f["tau"], tau_img[sl])
        nuc_labels[sl] = np.where(win, f["nuc"] * n, nuc_labels[sl])
        geoms[n], slices[n], centres_w[n], fits[n] = g, sl, cw, f["fit"]
        
    # keep only the piece holding the seed; a two-piece "cell" is a wrong annotation
    orphan = 0
    for n, slc in enumerate(ndi.find_objects(labels), start=1):
        if slc is None:
            continue
        m = labels[slc] == n
        cc, k = ndi.label(m)
        if k > 1:
            main = 1 + int(np.argmax(np.bincount(cc.ravel())[1:]))
            drop = m & (cc != main)
            labels[slc][drop] = 0
            nuc_labels[slc][drop] = 0
            tau_img[slc][drop] = 0.0
            orphan += int(drop.sum())
    
    present = sorted(set(np.unique(labels)) - {0})
    cvox = tape["xyz"][keep]
    tree = cKDTree(cvox)
    neigh = {n: [j + 1 for j in tree.query_ball_point(cvox[n - 1], TG.NEIGH_RADIUS.v)
                 if j + 1 != n] for n in present}
    neigh = {n: [j for j in v if j in neigh] for n, v in neigh.items()}

    info = dict(labels_present=present, cand_idx={n: int(keep[n - 1]) for n in present},
        geoms=geoms, slices=slices, centres_w=centres_w, fits=fits,
        centres_vox={n: cvox[n - 1] for n in present},
        neighbours=neigh, support=sup, orphan_vox=orphan,
        packing=float((labels > 0).sum() / max(sup.sum(), 1)))
    
    types = assign_types(info, tape, CellTypes, Fractions, rule=rule)
    
    names = list(Panel.Markers)
    out = {m: np.zeros(shape, np.float32) for m in names}

    for n in info["labels_present"]:
        ctype = CellTypes[types[n]]
        g, sl = info["geoms"][n], info["slices"][n]
        win = labels[sl] == n
        if not win.any():
            continue
        # the packed body, so a marker fills the claimed territory, not just the free shape
        f = stamp_cell(tape, info["cand_idx"][n], g, grid, sl, info["centres_w"][n],
                          grow=TG.GROW.v, L=L, l_min=l_min, fit=info["fits"][n])
        ptape = TapeDict(texture_noise=tape["texture_noise"][(slice(None),) + sl],
                         gate_noise=tape["gate_noise"][(slice(None),) + sl])
        off = 0
        for name, marker in Panel.Markers.items():
            lvl = ctype.level(name)
            if lvl > 0.0:
                img = render_marker(tape=ptape, marker=marker, cell=f, spacing=spacing, geom=g,
                                    um_per_vox=um_per_vox, edge_softness=0.0, pool_offset=off,
                                    pool_bank=pool_bank)
                # the level multiplies the rendered volume, which is exactly scaling amp
                if lvl != 1.0:
                    img = img * np.float32(lvl)
                out[name][sl] = np.where(win, img, out[name][sl])
            off += len(marker.noise_components)
    
    return out, labels, nuc_labels, tau_img, types, info


def build_tissue_V2(tape, ARCH, TG, shape, base_geom, spacing, Panel, CellTypes, Fractions,
                 um_per_vox, L=4, l_min=2, rule=None, pool_bank=None):
    """Tissue in three phases: TYPE ASSIGNMENT, then geometry, then appearance."""
    import tissue_structures as ts
    
    sup, field, report = ts.build_architecture(tape, ARCH, TG, shape, spacing, um_per_vox,
                                                   L=L, l_min=l_min)
    psi, thr = field["psi"], field["thr"]
    nz, ny, nx = shape

    fc_all = None
    if field is not None and ARCH.Structures:
        fc_all = tdir.sample_field(field, tape["xyz"][:, 0], tape["xyz"][:, 1], tape["xyz"][:, 2])
    if fc_all is not None:
        log_dens = np.array([np.log(max(float(st.DENSITY.v), 1e-6)) for st in ARCH.Structures])
        tilt = log_dens @ fc_all["m"]                            # (n_cand,), background = 0
        radii = TG.MIN_DIST.v * np.exp(-(1.0 / 3.0) * tilt)
        keep = thin_density(tape["xyz"], tape["order"], radii)
    else:
        keep = thin(tape["xyz"], tape["order"], TG.MIN_DIST.v)
    cx, cy, cz = tape["xyz"][keep].T
    keep = keep[sup[np.clip(cz.astype(int), 0, nz - 1),
                    np.clip(cy.astype(int), 0, ny - 1),
                    np.clip(cx.astype(int), 0, nx - 1)]]

    cvox_all = tape["xyz"][keep]

    if field is not None and ARCH.Structures and len(keep):
        cxs, cys, czs = cvox_all.T
        surv_lbl = field["labels"][np.clip(czs.astype(int), 0, nz - 1),
                                    np.clip(cys.astype(int), 0, ny - 1),
                                    np.clip(cxs.astype(int), 0, nx - 1)]
        for i, name in enumerate(field["struct_names"]):
            d = report["density"][name]
            pts_i = cvox_all[surv_lbl == i]
            d["survivors"] = int(len(pts_i))
            if len(pts_i) >= 2:
                d["nn_mean_vox"] = float(cKDTree(pts_i).query(pts_i, k=2)[0][:, 1].mean())
            else:
                d["nn_mean_vox"] = None
            d["starved"] = bool(d["requested"] > 1.0 and d["raw_in_territory"] > 0
                                 and d["survivors"] / d["raw_in_territory"] > 0.9)

    tree_all = cKDTree(cvox_all)
    lbl_all = list(range(1, len(keep) + 1))
    neigh_all = {n: [j + 1 for j in tree_all.query_ball_point(cvox_all[n - 1], TG.NEIGH_RADIUS.v)
                     if j + 1 != n] for n in lbl_all}
    pre = dict(labels_present=lbl_all, cand_idx={n: int(keep[n - 1]) for n in lbl_all},
               centres_vox={n: cvox_all[n - 1] for n in lbl_all}, neighbours=neigh_all,
               field=field)
    feat = None
    if field is not None and len(keep):
        # ONE vectorised read for every cell centre, shared by typing and the geometry pass. A
        # 3-D Q needs six interpolations per cell plus one per structure, so sampling per cell in
        # two separate places would be thousands of tiny map_coordinates calls at the cell counts
        # a small MIN_DIST produces.
        fc = ({k: v[..., keep] for k, v in fc_all.items()} if fc_all is not None
              else tdir.sample_field(field, cvox_all[:, 0], cvox_all[:, 1], cvox_all[:, 2]))
        feat_names = list(field.get("features", {}))          # e.g. ["S", "ring"]
        feat = {n: {"n": fc["n"][:, n - 1], "theta": float(fc["theta"][n - 1]),
                    "m": fc["m"][:, n - 1],
                    **{k: float(fc[k][n - 1]) for k in feat_names}}
                for n in lbl_all}
        pre["feat"] = feat
        pre["log_weights"] = np.stack([st.log_weights(list(CellTypes))
                                       for st in ARCH.Structures]) if ARCH.Structures else None
        # centre every FIELD feature against the population actually being typed, so a weight
        # moves cells between compartments instead of changing how many of that type there are.
        # The profile term is deliberately not centred -- see assign_types.
        pre["feat_refs"] = {k: float(fc[k].mean()) for k in feat_names}
    # every thinned candidate is typed; the return value is filtered to the survivors below
    types_all = assign_types(pre, tape, CellTypes, Fractions, rule=rule)

    # how much the local architecture wants cells to follow the director at all: 1 outside every
    # structure (the flow is simply what it is), each structure's own ALIGN inside it. ALIGN
    # defaults to 1.0, so this is exactly 1 everywhere until someone turns one down.
    s_align = None
    if feat is not None and ARCH.Structures:
        av = np.array([1.0 - float(st.ALIGN.v) for st in ARCH.Structures])

        def s_align(fn):
            return float(np.clip(1.0 - float(fn["m"] @ av), 0.0, 1.0))

    grid = centred_grid_3d(shape, spacing)
    best = np.full(shape, np.inf, np.float32)
    labels = np.zeros(shape, np.int32)
    nuc_labels = np.zeros(shape, np.int32)
    tau_img = np.zeros(shape, np.float32)
    geoms, slices, centres_w, fits = {}, {}, {}, {}
    g_align = float(ARCH.ALIGN.v) if ARCH is not None else 0.0

    for n, i in enumerate(keep, start=1):
        # a type owns its biology; base_geom is the fallback for a type that carries none
        ct = CellTypes[types_all[n]]
        tgeom = ct.Geometry if getattr(ct, "Geometry", None) is not None else base_geom
        if tgeom is None:
            raise ValueError(
                f"cell type {types_all[n]!r} has no Geometry and no base_geom was given. "
                f"A type owns its own shape now -- give it one, or pass base_geom as a fallback.")
        fn = feat[n] if feat is not None else None
        g = cell_geometry(tgeom, tape, i, TG,
                          n_dir=(fn["n"] if fn is not None else None),
                          S=(fn["S"] if fn is not None else 0.0),
                          align=g_align * float(tgeom.ALIGN.v)
                          * (s_align(fn) if s_align is not None else 1.0))
        sl = patch_grid(tape["xyz"][i],
                         reach_px(R=g.RADIUS.v, rough=g.ROUGH.v, elong=g.ELONG.v, grow=TG.GROW.v), 
                         shape)
        if any(s.stop - s.start <= 0 for s in sl):
            continue
        cw = vox_to_world(tape["xyz"][i], shape, spacing)
        f = stamp_cell(tape, i, g, grid, sl, cw, L = L, l_min = l_min, grow=1.0)

        # tesselation to decide which cell is closest to each pixel 
        win = (f["d"] < best[sl]) & (f["d"] <= TG.GROW.v) & sup[sl]
        best[sl] = np.where(win, f["d"], best[sl])
        labels[sl] = np.where(win, n, labels[sl])
        tau_img[sl] = np.where(win, f["tau"], tau_img[sl])
        nuc_labels[sl] = np.where(win, f["nuc"] * n, nuc_labels[sl])
        geoms[n], slices[n], centres_w[n], fits[n] = g, sl, cw, f["fit"]
        
    # keep only the piece holding the seed; a two-piece "cell" is a wrong annotation
    orphan = 0
    for n, slc in enumerate(ndi.find_objects(labels), start=1):
        if slc is None:
            continue
        m = labels[slc] == n
        cc, k = ndi.label(m)
        if k > 1:
            main = 1 + int(np.argmax(np.bincount(cc.ravel())[1:]))
            drop = m & (cc != main)
            labels[slc][drop] = 0
            nuc_labels[slc][drop] = 0
            tau_img[slc][drop] = 0.0
            orphan += int(drop.sum())
    
    present = sorted(set(np.unique(labels)) - {0})
    cvox = tape["xyz"][keep]
    tree = cKDTree(cvox)
    neigh = {n: [j + 1 for j in tree.query_ball_point(cvox[n - 1], TG.NEIGH_RADIUS.v)
                 if j + 1 != n] for n in present}
    neigh = {n: [j for j in v if j in neigh] for n, v in neigh.items()}

    info = dict(labels_present=present, cand_idx={n: int(keep[n - 1]) for n in present},
        geoms=geoms, slices=slices, centres_w=centres_w, fits=fits,
        centres_vox={n: cvox[n - 1] for n in present},
        neighbours=neigh, support=sup, orphan_vox=orphan,
        packing=float((labels > 0).sum() / max(sup.sum(), 1)),
        # the field, the potential behind it, and the ledger's realised fractions, so a caller
        # can plot or report them. info["support"] already excludes every lumen, and that is what
        # artifacts consume, so debris and detachment avoid the lumina with no artifact-side
        # change at all.
        field=field, psi=psi, psi_bar=psi.mean(0), thr=thr, report=report, arch=ARCH)

    # typed before the geometry pass; filtered to the survivors here so the returned dict
    # still means "the type of every cell in the label image", exactly as it always did
    types = {n: types_all[n] for n in present}

    names = list(Panel.Markers)
    out = {m: np.zeros(shape, np.float32) for m in names}

    for n in info["labels_present"]:
        ctype = CellTypes[types[n]]
        g, sl = info["geoms"][n], info["slices"][n]
        win = labels[sl] == n
        if not win.any():
            continue
        # the packed body, so a marker fills the claimed territory, not just the free shape
        f = stamp_cell(tape, info["cand_idx"][n], g, grid, sl, info["centres_w"][n],
                          grow=TG.GROW.v, L=L, l_min=l_min, fit=info["fits"][n])
        ptape = TapeDict(texture_noise=tape["texture_noise"][(slice(None),) + sl],
                         gate_noise=tape["gate_noise"][(slice(None),) + sl])
        off = 0
        for name, marker in Panel.Markers.items():
            lvl = ctype.level(name)
            if lvl > 0.0:
                img = render_marker(tape=ptape, marker=marker, cell=f, spacing=spacing, geom=g,
                                    um_per_vox=um_per_vox, edge_softness=0.0, pool_offset=off,
                                    pool_bank=pool_bank)
                # the level multiplies the rendered volume, which is exactly scaling amp
                if lvl != 1.0:
                    img = img * np.float32(lvl)
                out[name][sl] = np.where(win, img, out[name][sl])
            off += len(marker.noise_components)
    
    return out, labels, nuc_labels, tau_img, types, info

