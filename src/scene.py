from collections import namedtuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import config as cfg
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
    """Volume-preserving ellipsoid semi-axes for an aspect ratio `elong` = LONG : SHORT.
    A = elong^(2/3) along body-z, B = elong^(-1/3) across, so A / B = elong exactly and
    A * B^2 = 1 always: elongation cannot leak into cell size.

    RENAMED MEANING (2026-11-16): this used to return (elong, elong^-1/2), so the realised
    long:short ratio was elong^1.5 -- ELONG 7 was a 18.5:1 needle and the wide prior's median type
    (ELONG ~6) had a cross-section radius of 0.41 R. Measured on the run13 eval set: median cell
    section diameter 4.7 um, smaller than a real nucleus. A stored ELONG from before that date
    means elong^1.5 in the new units.
    """
    e = float(elong)
    A = e ** (2.0 / 3.0)
    B = e ** (-1.0 / 3.0)
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


def patch_grid(centre_xyz, reach, shape, spacing=None):
    """Box of half-width `reach` around a centre, clipped to the volume. `reach` is in LATERAL
    voxels; with `spacing` (sz, sy, sx) it is converted to plane indices per axis, so a centre that
    sits outside the block in z (a cell cut far from its middle) gets an EMPTY z-slice when its body
    cannot reach the block, instead of the whole stack. spacing=None is the old isotropic box."""
    nz, ny, nx = shape
    cx, cy, cz = centre_xyz
    if spacing is None:
        rz = ry = rx = reach
    else:
        sz, sy, sx = spacing
        rz, ry, rx = (int(np.ceil(reach / max(float(s), 1e-6))) + 1 for s in (sz, sy, sx))
    cz, cy, cx = int(np.floor(cz)), int(np.floor(cy)), int(np.floor(cx))
    return (slice(max(0, cz - rz), min(nz, cz + rz + 1)),
            slice(max(0, cy - ry), min(ny, cy + ry + 1)),
            slice(max(0, cx - rx), min(nx, cx + rx + 1)))


def reach_px(R, rough, elong = 1.0, grow=1.0, pad=2):
    """Furthest pixel this cell can claim, in image vox: the longest semi-axis of the stretched
    body (see _stretch), grown, with 3 SD of roughness on top."""
    return int(np.ceil(R * grow * float(_stretch(elong).max()) * np.exp(3.0 * rough))) + pad


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
    # the clipped nuclear signed distance and the in-nucleus radial fraction, so a caller can build
    # tau against a DIFFERENT outer surface (the realised territory, see territory_tau) while the
    # nuclear edge stays this exact curve
    out["psi_n"] = psi_n
    out["nuc_r"] = rho_n / np.maximum(r_n, 1e-6)
    return out


def territory_edge_distance(labels, spacing):
    """Distance (lateral voxels) from every voxel to the nearest CELL BOUNDARY actually realised in
    the packed tissue: a voxel is a boundary voxel if any of its six neighbours carries another
    label (another cell, or the gap), and gets distance 0.

    This is where the plasma membrane really is. The grown contour (r_c x GROW) is only where the
    cell WOULD end if nothing stopped it; in confluent tissue a neighbour stops it long before, so
    a marker localised at tau = 1 on the grown contour lit nothing at cell-cell contacts and
    same-type neighbours were separated by their nuclei alone -- the model then learned "cell =
    region around a nucleus". The block faces are NOT boundaries: a cell cut by the edge of the
    volume continues outside it, so no membrane is drawn along the tile border. One EDT for the
    whole block (~10 ms); every cell reads its own patch of it."""
    edge = np.zeros(labels.shape, bool)
    for ax in range(labels.ndim):
        a = [slice(None)] * labels.ndim
        b = [slice(None)] * labels.ndim
        a[ax], b[ax] = slice(1, None), slice(None, -1)
        diff = labels[tuple(a)] != labels[tuple(b)]
        edge[tuple(a)] |= diff
        edge[tuple(b)] |= diff
    if not edge.any():
        return np.full(labels.shape, 1e6, np.float32)
    return ndi.distance_transform_edt(~edge, sampling=spacing).astype(np.float32)


def territory_tau(f, edge_dist):
    """tau against the REALISED territory: inside the nucleus the usual radial fraction - 1 (so
    -1 at the centre, 0 at the envelope); outside it the fraction of the way from the nuclear
    envelope to the nearest territory boundary, dn / (dn + de) -- 0 at the envelope, exactly 1 on
    the boundary voxels, wherever the neighbours put them. `f` is a cell_fields dict stamped at
    grow = 1 (so its nuclear edge is the labelled one); `edge_dist` the matching patch of
    territory_edge_distance."""
    psi_n = f["psi_n"]
    dn = np.maximum(psi_n, 0.0)
    return np.where(psi_n < 0.0, f["nuc_r"] - 1.0,
                    dn / np.maximum(dn + edge_dist, 1e-6)).astype(np.float32)


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

def nuc_elong_fit(nuc_elong, elong, nuc_frac, slack=0.85):
    """The nucleus's own aspect ratio, pulled just far enough toward the cell's that the stretched
    nucleus still fits inside the stretched cell. Both bodies are volume-preserving ellipsoid
    stretches of their star shapes along the SAME axis, so a nucleus more elongated than its cell
    pokes out at the poles by (e_n / E)^(2/3), and a rounder one out of the flanks by
    (E / e_n)^(1/3); the containment fit in `_cell_fit` compares the two un-stretched radii and
    cannot see either. With c = slack / nuc_frac the room available, the admissible range is
    [E / c^3, E x c^1.5]: at nuc_frac 0.25 that is E/39 .. 6.3 E (free), at nuc_frac 0.7 only
    E/1.8 .. 1.3 E -- a nucleus that fills its cell must share its shape. A smooth clamp of the
    declared value, never a redraw."""
    c = max(float(slack) / max(float(nuc_frac), 1e-6), 1.0)
    return float(np.clip(nuc_elong, elong / c ** 3, elong * c ** 1.5))


def cell_geometry(base, ttape, i, tg, n_dir=None, S=0.0, align=0.0):
    """This cell's own geometry: the type's shape plus its frozen size / orientation draws.

    The long axis starts ISOTROPIC on the sphere (cos(polar) uniform; it used to be uniform in the
    polar ANGLE, which piles axes up along +-z, so twice as many cells as there should be were seen
    end-on). It is then turned toward the tile's grain (TissueGeometry.AXIS_*, by AXIS_ALIGN x the
    type's ALIGN) and, inside a structure, on toward the local director.
    """
    r = float(np.clip(base.RADIUS.v * np.exp(tg.SIZE_SIGMA.v * ttape["z_size"][i]),
                      base.RADIUS.lo, base.RADIUS.hi))
    nf = float(np.clip(base.NUC_FRAC.v * np.exp(tg.NUC_FRAC_SIGMA.v * ttape["z_nucfrac"][i]),
                       base.NUC_FRAC.lo, base.NUC_FRAC.hi))
    o = ttape["u_orient3"][i]

    p, z = np.arccos(np.clip(1.0 - 2.0 * float(o[0]), -1.0, 1.0)), 2.0 * np.pi * float(o[1])
    d = np.array([np.sin(p) * np.cos(z), np.sin(p) * np.sin(z), np.cos(p)])
    t_align = float(getattr(tg, "AXIS_ALIGN", None).v) * float(base.ALIGN.v) \
        if getattr(tg, "AXIS_ALIGN", None) is not None else 0.0
    if t_align > 0.0:
        tp, tz = np.deg2rad(float(tg.AXIS_POLAR_DEG.v)), np.deg2rad(float(tg.AXIS_AZIM_DEG.v))
        d = tdir.slerp_axis(d, np.array([np.sin(tp) * np.cos(tz), np.sin(tp) * np.sin(tz),
                                         np.cos(tp)]), t_align)
    a = float(align) * float(S)
    if n_dir is not None and a > 0.0:
        d = tdir.slerp_axis(d, n_dir, a)
    polar = float(np.rad2deg(np.arccos(np.clip(d[2], -1.0, 1.0))))
    azim = float(np.rad2deg(np.arctan2(d[1], d[0])) % 360.0)
    ne = float(np.clip(nuc_elong_fit(base.NUC_ELONG.v, base.ELONG.v, nf),
                       base.NUC_ELONG.lo, base.NUC_ELONG.hi))
    return dataclasses.replace(base, RADIUS=r, NUC_FRAC=nf, NUC_ELONG=ne, POLAR_DEG=float(polar),
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
    # cover=0 is the empty-tile null: the quantile at 1.0 would otherwise keep the single max voxel,
    # so an honest zero-cell tile needs an explicit all-False support.
    if float(cover) <= 0.0:
        return np.zeros(z.shape, bool)
    return z >= np.quantile(z, 1.0 - float(np.clip(cover, 0.0, 1.0)))


def assign_types(info, tape, CellTypes, fractions, rule=None, compositions=None):
    """Decide each cell's type.

    Two priors live here, chosen by `compositions`:

    - FIXED mode (`compositions is None`, the default): the global Fractions, tilted by the
      structure Profile the cell sits in and by its cell type's Affinity for each named tissue
      feature. Unchanged from before.
    - FLEXIBLE mode (plan 3.4): a per-context composition replaces the Profile log-weights.
      `p(type k | x) = sum_c w_c(x) . pi_c[k]`, where `w_c` is the cell's membership of context c
      (contexts are [background] + the distinct structure KINDS, not per-instance), and `pi` is
      the composition matrix in `CellTypes` order. There is no free additive direction here, so
      nothing about the typing is hidden from the image. Affinity/Fractions are not consulted.

    Both sample the chosen type with the SAME frozen per-cell uniform `tape["u_type"]`, so moving
    a weight or a composition entry moves cells between types without resampling anything.
    """
    type_names = list(CellTypes)
    f = np.asarray(fractions, float)
    flat = np.cumsum(f / np.sum(f))
    field = info.get("field")
    fmap = info.get("feat")
    lw = info.get("log_weights")
    refs = info.get("feat_refs", {})
    feat_names = list((field or {}).get("features", {})) if field is not None else []
    if compositions is not None:
        pi = np.asarray(compositions["pi"], float)                 # (C, K) in type_names order
        s2c = np.asarray(compositions["struct_to_ctx"], int)       # per structure -> context idx
        n_ctx = pi.shape[0]
        if compositions["type_order"] != type_names:
            raise ValueError(
                "compositions['type_order'] must match list(CellTypes) exactly; got "
                f"{compositions['type_order']} vs {type_names}. Build pi in CellTypes order.")
    types = {}
    for n in info["labels_present"]:
        i = info["cand_idx"][n]
        feat = {}
        if compositions is not None:
            # a structure-free tile is ALL background context, so its cells follow pi[0]. This
            # branch used to sit behind `field is None -> flat`, so every structure-free tile
            # (a quarter of the prior) was typed uniformly while theta said Dirichlet(pi[0]).
            w = np.zeros(n_ctx)
            if field is not None and fmap is not None:
                feat = fmap[n]
                m = np.asarray(feat["m"], float)                   # membership per structure
                np.add.at(w, s2c, m)                               # pool instances into their kind
                w[0] += max(0.0, 1.0 - float(m.sum()))             # the rest is background
            else:
                w[0] = 1.0
            w = w / (w.sum() + 1e-12)
            p = w @ pi
            cuts = np.cumsum(p / (p.sum() + 1e-12))
        elif field is None or fmap is None:
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
        base = type_names[min(int(np.searchsorted(cuts, tape["u_type"][i])), len(type_names) - 1)]
        ctx = CellContext(label=n, index=i, centre=info["centres_vox"][n],
                          neigh_labels=info["neighbours"][n], types=types, feat=feat)
        types[int(n)] = base if rule is None else rule(ctx, base)
    return types

class TapeDict(dict):
    """dict that also allows attribute access, so it works wherever a Tape does."""
    __getattr__ = dict.__getitem__

def seed_positions(tape, TG, shape, spacing):
    """Candidate cell centres as (x, y, z) plane INDICES, with z stretched over the seeded slab.

    The block is only the few planes the section is cut from; the tissue it is cut OUT of is much
    thicker. The tape's frozen z draw (uniform over the block) is therefore mapped, linearly, onto
    a slab TG.SLAB lateral-voxels thick centred on the block, so most centres sit ABOVE or BELOW
    the block and their cells enter it as caps and off-centre cuts -- including the profiles that
    show cytoplasm but no nucleus, which are the majority for any cell much bigger than the
    section (a 20 um cardiomyocyte cut by a 4 um section shows its nucleus about one time in
    three). Until 2026-11-16 every centre sat inside the 7.8 um block, so every cell bigger than
    that showed its nucleus and the model learned that a cell IS the region around a nucleus.
    A z index outside [0, nz) is legal and means exactly that. SLAB = nz x sz reproduces the old
    seeding up to a half-plane shift. Returns (xyz index coords, physical coords in lateral voxels)."""
    nz = shape[0]
    sz, sy, sx = spacing
    xyz = np.array(tape["xyz"], float)
    slab = float(TG.SLAB.v) if getattr(TG, "SLAB", None) is not None else nz * sz
    xyz[:, 2] = (xyz[:, 2] / nz - 0.5) * slab / sz + (nz - 1) / 2.0
    return xyz, xyz * np.array([sx, sy, sz], float)


def grain_metric(pts_w, TG, CellTypes):
    """Candidate positions in the metric the hard-core thinning should use when the tile has a
    grain. Parallel rods cannot sit closer side by side than their width, but must be allowed to
    sit a whole length apart end to end: the exclusion zone is the cell's own ellipsoid, not a
    sphere. So space is squeezed by E^(2/3) along the tile axis and stretched by E^(1/3) across it
    (volume-preserving, like scene._stretch) and the usual spherical thinning then excludes an
    ellipsoid of aspect E. E is the geometric mean of ELONG^(AXIS_ALIGN x type ALIGN) over the
    types: a type that ignores the grain, or a tile without one, contributes exactly 1.
    Without this, rods aligned THROUGH the section were thinned as spheres of their equivalent
    radius -- measured: 442 cross-sections of 2.8 um in one tile, each rod overlapping a dozen
    others along its length."""
    a = float(TG.AXIS_ALIGN.v) if getattr(TG, "AXIS_ALIGN", None) is not None else 0.0
    if a <= 0.0 or not CellTypes:
        return pts_w
    le = [a * float(ct.Geometry.ALIGN.v) * np.log(max(float(ct.Geometry.ELONG.v), 1.0))
          for ct in CellTypes.values() if getattr(ct, "Geometry", None) is not None]
    e = float(np.exp(np.mean(le))) if le else 1.0
    if e <= 1.0 + 1e-6:
        return pts_w
    tp, tz = np.deg2rad(float(TG.AXIS_POLAR_DEG.v)), np.deg2rad(float(TG.AXIS_AZIM_DEG.v))
    n = np.array([np.sin(tp) * np.cos(tz), np.sin(tp) * np.sin(tz), np.cos(tp)])
    along = pts_w @ n
    across = pts_w - along[:, None] * n
    return across * e ** (1.0 / 3.0) + (along * e ** (-2.0 / 3.0))[:, None] * n


def candidate_types(compositions, fc_all, u_type):
    """The type index every CANDIDATE would take, by exactly the rule assign_types applies to the
    survivors (same context membership, same frozen uniform) -- so a per-type spacing can be used
    for the thinning itself without the typing ever disagreeing with it afterwards."""
    pi = np.asarray(compositions["pi"], float)
    s2c = np.asarray(compositions["struct_to_ctx"], int)
    n = len(u_type)
    w = np.zeros((n, pi.shape[0]))
    if fc_all is not None and len(s2c):
        m = np.asarray(fc_all["m"], float)                       # (n_struct, n)
        for j, c in enumerate(s2c):
            w[:, c] += m[j]
        w[:, 0] += np.maximum(0.0, 1.0 - m.sum(0))
    else:
        w[:, 0] = 1.0
    w = w / (w.sum(1, keepdims=True) + 1e-12)
    pr = w @ pi
    cuts = np.cumsum(pr / (pr.sum(1, keepdims=True) + 1e-12), axis=1)
    k = (cuts < np.asarray(u_type, float)[:, None]).sum(1)
    return np.minimum(k, pi.shape[1] - 1)


def build_tissue_V2(tape, ARCH, TG, shape, base_geom, spacing, Panel, CellTypes, Fractions,
                 um_per_vox, L=4, l_min=2, rule=None, pool_bank=None, compositions=None,
                 background=None):
    """Tissue in three phases: TYPE ASSIGNMENT, then geometry, then appearance.

    `compositions` (plan 3.4, flexible mode) switches type assignment from the Fractions/Profile
    prior to a per-context composition; see `assign_types`. None keeps the old behaviour exactly.

    `background` is the tile's hidden shared signal (flexible_types.sample_background): markers no
    channel OWNS but every channel sees a share of. Each is rendered per cell like a real marker,
    at the cell type's own level, and the volumes come back under info["bg_vols"].
    """
    import tissue_structures as ts

    sup, field, report = ts.build_architecture(tape, ARCH, TG, shape, spacing, um_per_vox,
                                                   L=L, l_min=l_min)
    # field is None on the structure-free path (no director, uniform thinning)
    psi = field["psi"] if field is not None else None
    thr = field["thr"] if field is not None else None
    nz, ny, nx = shape
    xyz, pts_w = seed_positions(tape, TG, shape, spacing)

    fc_all = None
    if field is not None and ARCH.Structures:
        fc_all = tdir.sample_field(field, xyz[:, 0], xyz[:, 1], xyz[:, 2])
    radii = np.full(len(xyz), float(TG.MIN_DIST.v))
    if fc_all is not None:
        log_dens = np.array([np.log(max(float(st.DENSITY.v), 1e-6)) for st in ARCH.Structures])
        tilt = log_dens @ fc_all["m"]                            # (n_cand,), background = 0
        radii = radii * np.exp(-(1.0 / 3.0) * tilt)
    # spacing follows the cell's own size: a big type keeps its neighbours further away than a
    # small one, so a tile can hold 20 um cells with 6 um cells between them. With one spacing for
    # all (SIZE_COUPLE 0) every realised cell is about MIN_DIST across whatever its RADIUS says.
    couple = float(TG.SIZE_COUPLE.v) if getattr(TG, "SIZE_COUPLE", None) is not None else 0.0
    if couple > 0.0 and compositions is not None and len(CellTypes) > 1:
        k_all = candidate_types(compositions, fc_all, tape["u_type"])
        logr = np.log(np.array([max(float(ct.Geometry.RADIUS.v), 1e-6)
                                for ct in CellTypes.values()]))[k_all]
        # centred on the candidates' own mean, so MIN_DIST stays the TYPICAL spacing
        radii = radii * np.clip(np.exp(couple * (logr - logr.mean())), 0.35, 3.0)
    # hard-core thinning in PHYSICAL units: in plane indices the 4x coarser z let two centres
    # stack 1.3 um apart, which is what forced a monolayer
    keep = thin_density(grain_metric(pts_w, TG, CellTypes), tape["order"], radii)
    cx, cy, cz = xyz[keep].T
    keep = keep[sup[np.clip(np.floor(cz).astype(int), 0, nz - 1),
                    np.clip(cy.astype(int), 0, ny - 1),
                    np.clip(cx.astype(int), 0, nx - 1)]]

    cvox_all = xyz[keep]

    if field is not None and ARCH.Structures and len(keep):
        cxs, cys, czs = cvox_all.T
        surv_lbl = field["labels"][np.clip(np.floor(czs).astype(int), 0, nz - 1),
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

    tree_all = cKDTree(pts_w[keep]) if len(keep) else None
    lbl_all = list(range(1, len(keep) + 1))
    neigh_all = {n: [j + 1 for j in tree_all.query_ball_point(pts_w[keep[n - 1]], TG.NEIGH_RADIUS.v)
                     if j + 1 != n] for n in lbl_all}
    pre = dict(labels_present=lbl_all, cand_idx={n: int(keep[n - 1]) for n in lbl_all},
               centres_vox={n: cvox_all[n - 1] for n in lbl_all}, neighbours=neigh_all,
               field=field)
    feat = None
    if field is not None and ARCH.Structures and len(keep):
        # ONE vectorised read for every cell centre, shared by typing and the geometry pass. A
        # 3-D Q needs six interpolations per cell plus one per structure, so sampling per cell in
        # two separate places would be thousands of tiny map_coordinates calls at the cell counts
        # a small MIN_DIST produces.
        fc = {k: v[..., keep] for k, v in fc_all.items()}
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
    types_all = assign_types(pre, tape, CellTypes, Fractions, rule=rule, compositions=compositions)

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
        sl = patch_grid(xyz[i],
                         reach_px(R=g.RADIUS.v, rough=g.ROUGH.v, elong=g.ELONG.v, grow=TG.GROW.v),
                         shape, spacing=spacing)
        # a centre far above or below the block whose body cannot reach it: not in this section
        if any(s.stop - s.start <= 0 for s in sl):
            continue
        cw = vox_to_world(xyz[i], shape, spacing)
        f = stamp_cell(tape, i, g, grid, sl, cw, L = L, l_min = l_min, grow=1.0)

        # tesselation to decide which cell is closest to each pixel
        win = (f["d"] < best[sl]) & (f["d"] <= TG.GROW.v) & sup[sl]
        if not win.any():
            continue
        best[sl] = np.where(win, f["d"], best[sl])
        labels[sl] = np.where(win, n, labels[sl])
        nuc_labels[sl] = np.where(win, f["nuc"] * n, nuc_labels[sl])
        geoms[n], slices[n], centres_w[n], fits[n] = g, sl, cw, f["fit"]

    # keep only the piece holding the most volume; a two-piece "cell" is a wrong annotation
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
            orphan += int(drop.sum())

    present = sorted(set(np.unique(labels).tolist()) - {0})
    neigh = {n: neigh_all[n] for n in present}
    pset = set(present)
    neigh = {n: [j for j in v if j in pset] for n, v in neigh.items()}

    info = dict(labels_present=present, cand_idx={n: int(keep[n - 1]) for n in present},
        geoms=geoms, slices=slices, centres_w=centres_w, fits=fits,
        centres_vox={n: cvox_all[n - 1] for n in present},
        neighbours=neigh, support=sup, orphan_vox=orphan,
        packing=float((labels > 0).sum() / max(sup.sum(), 1)),
        # the field, the potential behind it, and the ledger's realised fractions, so a caller
        # can plot or report them. info["support"] already excludes every lumen, and that is what
        # artifacts consume, so debris and detachment avoid the lumina with no artifact-side
        # change at all. psi is None on the structure-free path.
        field=field, psi=psi, psi_bar=(psi.mean(0) if psi is not None else None),
        thr=thr, report=report, arch=ARCH)

    # typed before the geometry pass; filtered to the survivors here so the returned dict
    # still means "the type of every cell in the label image", exactly as it always did
    types = {n: types_all[n] for n in present}

    # --- appearance. The cell IS its realised territory: tau runs from the nuclear envelope to the
    # territory boundary (territory_tau), and brightness is normalised against an ideal whole cell
    # (render.loc_norm), never against the piece of it that happens to lie in the block.
    edge_dist = territory_edge_distance(labels, spacing)
    tau_img = np.zeros(shape, np.float32)
    names = list(Panel.Markers)
    out = {m: np.zeros(shape, np.float32) for m in names}
    bg_markers = list(background["markers"]) if background else []
    bg_vols = [np.zeros(shape, np.float32) for _ in bg_markers]
    z_expr = getattr(tape, "z_expr", None)
    boxes = ndi.find_objects(labels)

    for n in info["labels_present"]:
        ctype = CellTypes[types[n]]
        g = info["geoms"][n]
        # the territory's own bounding box, not the reach box: at GROW 3 the reach box is up to
        # 27x the territory, and every texture below is an FFT over the patch
        sl = boxes[n - 1]
        win = labels[sl] == n
        if not win.any():
            continue
        i_c = info["cand_idx"][n]
        f = dict(stamp_cell(tape, i_c, g, grid, sl, info["centres_w"][n],
                            grow=1.0, L=L, l_min=l_min, fit=info["fits"][n]))
        f["tau"] = territory_tau(f, edge_dist[sl])
        f["cell"] = win
        tau_img[sl] = np.where(win, f["tau"], tau_img[sl])
        ptape = TapeDict(texture_noise=tape["texture_noise"][(slice(None),) + sl],
                         gate_noise=tape["gate_noise"][(slice(None),) + sl], seed=tape.seed)
        nf = float(g.NUC_FRAC.v)
        r_short = float(g.RADIUS.v) * float(_stretch(g.ELONG.v).min())
        off = 0
        for c_idx, (name, marker) in enumerate(Panel.Markers.items()):
            lvl = ctype.level(name)
            if lvl > 0.0:
                # cell-to-cell spread of one type's expression: lognormal, mean-preserving, on a
                # frozen (candidate, channel) draw -- so a type's LEVEL stays its mean brightness
                cv = float(marker.cell_cv.v) if getattr(marker, "cell_cv", None) is not None else 0.0
                if cv > 0.0 and z_expr is not None:
                    lvl = lvl * float(np.exp(cv * z_expr[i_c, c_idx % z_expr.shape[1]] - 0.5 * cv * cv))
                img = render_marker(tape=ptape, marker=marker, cell=f, spacing=spacing, geom=g,
                                    um_per_vox=um_per_vox, edge_softness=0.0, pool_offset=off,
                                    pool_bank=pool_bank, nuc_frac=nf, radius_vox=r_short)
                # the level multiplies the rendered volume, which is exactly scaling amp
                if lvl != 1.0:
                    img = img * np.float32(lvl)
                out[name][sl] = np.where(win, img, out[name][sl])
            off += len(marker.noise_components)
        for j, marker in enumerate(bg_markers):
            lvl = float(background["type_level"][types[n]][j])
            if lvl > 0.0:
                cv = float(background["cell_cv"][j])
                if cv > 0.0 and z_expr is not None:
                    lvl = lvl * float(np.exp(cv * z_expr[i_c, (len(names) + j) % z_expr.shape[1]]
                                             - 0.5 * cv * cv))
                img = render_marker(tape=ptape, marker=marker, cell=f, spacing=spacing, geom=g,
                                    um_per_vox=um_per_vox, edge_softness=0.0, pool_offset=off,
                                    pool_bank=pool_bank, nuc_frac=nf, radius_vox=r_short)
                bg_vols[j][sl] = np.where(win, img * np.float32(lvl), bg_vols[j][sl])
            off += len(marker.noise_components)

    info["bg_vols"] = bg_vols
    info["edge_dist"] = edge_dist
    return out, labels, nuc_labels, tau_img, types, info


def artifact_priority(cell2d, art2d):
    """Artifacts occlude whatever cell sits underneath them."""
    out = cell2d.copy()
    out[art2d > 0] = art2d[art2d > 0]
    return out


def build_training_mask(combined2d, detach_gate, detach_threshold=0.5, whole_cell_frac=0.5,
                         min_area=50):
    """Cellpose-ready cell-only raster from a combined (cell+artifact, disjoint ids) 2D
    collapse: a cell is dropped -- as a WHOLE object, never partially erased -- if more than
    `whole_cell_frac` of its footprint is covered by an artifact or a sufficiently detached/
    lifted region, or if its area is under `min_area` pixels. Partial erasure was considered and
    rejected: it can split a single cell into two disconnected pieces under one id, which is
    worse training data than dropping it entirely.
    """
    cell2d_only = np.where(combined2d < cfg.ARTIFACT_ID0, combined2d, 0)
    exclude = (combined2d >= cfg.ARTIFACT_ID0) | (detach_gate > detach_threshold)
    out = cell2d_only.copy()
    for cid in np.unique(cell2d_only):
        if cid == 0:
            continue
        m = cell2d_only == cid
        if float(exclude[m].mean()) > whole_cell_frac or int(m.sum()) < min_area:
            out[m] = 0
    return out

