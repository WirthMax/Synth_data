import numpy as np
from scipy.spatial.transform import Rotation as R

from parameter import CellContext
from render import render_marker

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
    # compute w with the spherical harmonics
    
    # compute the weight for each degree, such that all weights of a degree are equal
    w = sh_degrees(L, l_min).astype(float) ** -float(beta)
    # renormalization factor to ensure the area is correct
    # With this, changing beta only changes the kind of roughness, not the overall size of the cell.
    w = w * (kappa * np.sqrt(4.0 * np.pi) / np.sqrt((w ** 2).sum() + 1e-30))
    c = w*coeff
    
    # Compute quadrate to pinpount size and then scale accordingly
    uq, wq = _quad()
    gq = sh_basis(uq, L, l_min) @ c
    # volume at scale = 1
    vol = float((np.exp(3.0 * gq) * wq).sum() / 3.0)          
    scale = R * ((4.0 * np.pi / 3.0) / max(vol, 1e-30)) ** (1.0 / 3.0)
    return lambda u: scale * np.exp(sh_eval(u, c, L, l_min))

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
    # 1 Center to the cells coordinate systen
    d = (grid - np.asarray(centre, np.float32)).astype(np.float32, copy=False)
    # 2 Rotate so the long axis in in body-z
    if rot is not None:
        d = d @ np.asarray(rot, np.float32)
    # 3 Unstretch ellipsoid
    d = d / _stretch(elong).astype(np.float32)
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


def reach_px(R, elong, rough, grow=1.0, pad=2):
    """Furthest pixel this cell can claim, in image vox"""
    s = np.sqrt(elong)
    return int(np.ceil(R * grow * max(s, 1.0 / s) * np.exp(3.0 * rough))) + pad


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
        base = r_nuc_fn(dirs)[:, None] * dirs
    if rim_eff is None:
        rim_eff = rim * _rim_scale(r_cell_fn, dirs) if euclid_rim else rim
    p = np.asarray(delta, float) * np.asarray(off_hat, float) + base
    rho = np.linalg.norm(p, axis=-1)
    return rho - r_cell_fn(p / np.maximum(rho, 1e-30)[..., None]) + rim_eff
    
def _rim_scale(r_cell_fn, dirs, eps=1e-4):
    e = np.eye(3) * eps
    p = np.concatenate([dirs[None] + e[:, None, :], dirs[None] - e[:, None, :]])   # (6, n, 3)
    p /= np.linalg.norm(p, axis=-1, keepdims=True)
    lg = np.log(np.maximum(r_cell_fn(p), 1e-30))                                   # (6, n)
    d = (lg[:3] - lg[3:]) / (2 * eps)
    return np.sqrt(1.0 + (d * d).sum(axis=0))

def nuc_offset_budget(r_cell_fn, r_nuc_fn, off_hat, rim=1.5,
                      n_scan=33, n_bisect=25, euclid_rim=True):
    """
    """
    dirs = fit_directions(n_theta=24, n_phi=48)
    off_hat = np.asarray(off_hat, float)
    base = r_nuc_fn(dirs)[:, None] * dirs
    rim_eff = rim * _rim_scale(r_cell_fn, dirs) if euclid_rim else rim

    def worst(delta):
        return containment_residual(r_cell_fn, r_nuc_fn, off_hat, delta, rim, dirs,
                                    rim_eff=rim_eff, base=base).max()

    hi = max(float(np.max(r_cell_fn(dirs))) - float(np.min(np.atleast_1d(rim_eff))), 0.0)
    if hi <= 0.0:
        return 0.0
    grid = np.linspace(0.0, hi, n_scan)
    vals = np.array([worst(d) for d in grid])
    over = np.flatnonzero(vals > 0.0)
    if over.size == 0:
        return float(hi)                      # fits at any offset (near-spherical cell)
    k = int(over[0])
    if k == 0:
        return 0.0                            # s0 < 1: infeasible even centred
    lo, up = float(grid[k - 1]), float(grid[k])
    for _ in range(n_bisect):
        mid = 0.5 * (lo + up)
        lo, up = (lo, mid) if worst(mid) > 0.0 else (mid, up)
    return lo


def nuc_fit_scale(r_cell_fn, r_nuc_fn, rim=1.5, dirs=None, euclid_rim=True):
    """s0 = min_w (r_c(w) - rim_eff(w)) / r_n(w).

    s0 >= 1 means the CENTRED nucleus already fits with the full rim. s0 < 1 means `rough`,
    `nuc_frac` and `rim` are jointly infeasible for this cell and the clip in `cell_fields_3d`
    will bite at EVERY offset, including 0 -- no choice of `nuc_offset` can rescue it.

    Measured at radius = 24, rim = 1.5: at rough = 0.25 this affects 32% of cells for
    nuc_frac = 0.45, but 0% for nuc_frac <= 0.30.
    """
    dirs = fit_directions() if dirs is None else dirs
    rim_eff = rim * _rim_scale(r_cell_fn, dirs) if euclid_rim else rim
    return float(np.min((r_cell_fn(dirs) - rim_eff) / np.maximum(r_nuc_fn(dirs), 1e-30)))


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


def _cell_and_nucleus(c_cell, c_nuc, grid, L, l_min, nuc_corr, nuc_frac, radius, 
                      rough, beta, rot, elong, rim, nuc_offset, off_dir, off_mag, grow, 
                      centre = (0.0, 0.0, 0.0), rough_nuc = None, beta_nuc=None,
                      euclid_rim=True, nuc_fit_floor=None):
    """Shared core: both wrappers do exactly this, only the grid differs.

    body_grow scales the cell contour (not the nucleus); see cell_fields. Default 1.0 = the
    free shape used everywhere for the mask; > 1 gives the packed body for marker rendering.
    """
    cn = mix_harmonics(c_cell, c_nuc, nuc_corr)
    r_cell_fn = make_boundary(coeff = c_cell, R = radius, kappa = rough, 
                              beta = beta, L = L, l_min = l_min)
    if rough_nuc is None:
        rough_nuc = rough * 0.6
    if rough_nuc is None:
        beta_nuc = beta
    r_nuc_fn = make_boundary(coeff = cn, R = radius * nuc_frac, kappa = rough_nuc, 
                             beta = beta_nuc, L=L, l_min = l_min)
    rho_c, phi_c = body_frame(grid = grid, rot = rot, centre = centre, elong=elong)
    
    off = np.asarray(off_dir, float)
    off = off / (np.linalg.norm(off) + 1e-30)
    
    s0 = nuc_fit_scale(r_cell_fn, r_nuc_fn, rim, euclid_rim=euclid_rim)
    if nuc_fit_floor is not None and s0 < 1.0:
        # Opt-in rescue: shrink the nucleus uniformly. Smooth and isotropic, whereas the clip in
        # `cell_fields_3d` would instead cut a flat facet. Off by default because silently
        # shrinking nuclei would break what `nuc_frac` means.
        _r_nuc_raw, _s = r_nuc_fn, max(float(s0), float(nuc_fit_floor))
        r_nuc_fn = (lambda u, _f=_r_nuc_raw, _k=_s: _k * _f(u))
        
    # Compute freespace to membrane with rim space
    free = nuc_offset_budget(r_cell_fn, r_nuc_fn, off, rim=rim,
                        n_scan=33, n_bisect=25, euclid_rim=True)
    ncentre =  np.asarray(centre, float) + to_world(off * (nuc_offset * free * off_mag), rot, elong)
      
    rho_n, phi_n = body_frame(grid, rot = rot, centre = ncentre, elong=elong)

    f = cell_fields(rho_c, phi_c, r_cell_fn(phi_c), 
                    rho_n, phi_n, r_nuc_fn(phi_n), 
                    rim, grow=grow)
    h = containment_residual(r_cell_fn, r_nuc_fn, off, nuc_offset * free * off_mag, rim,
                             dirs=_quad()[0], euclid_rim=euclid_rim)
    f["nuc_centre"] = ncentre
    f["r_cell_fn"], f["r_nuc_fn"] = r_cell_fn, r_nuc_fn
    f["nuc_room"] = float(free)
    f["nuc_fit_scale"] = float(s0)
    f["nuc_shaved_frac"] = float((h > 0.0).mean())
    f["nuc_shaved_max"] = float(h.max())
    return f

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
               grow=1.0):
    """Tissue: candidate i stamped at (cy, cx, cz) on a local patch.

    `grow` sizes the cell contour only; `body_grow` scales the rendered cell contour (see
    cell_fields). They are set together (body_grow=grow) to render markers on the packed body.
    """
    return _cell_and_nucleus(
        c_cell=tape["sh"][i], c_nuc=tape["sh2"][i], grid=grid[sl], L=L, l_min=l_min,
        nuc_corr=geom.NUC_CORR.v, nuc_frac=geom.NUC_FRAC.v, radius=geom.RADIUS.v,
        rough=geom.ROUGH.v, beta=geom.BETA.v,
        rot=rotation_matrix(geom.POLAR_DEG.v, geom.AZIM_DEG.v, geom.ROLL_DEG.v),
        elong=geom.ELONG.v, rim=geom.RIM.v, nuc_offset=geom.NUC_OFFSET.v,
        off_dir=tape["u_offdir"][i], off_mag=tape["u_offmag"][i],
        grow=grow, centre=centre_world,
        rough_nuc=geom.NUC_ROUGH.v, beta_nuc=geom.NUC_BETA.v)


def vox_to_world(pts_xyz, shape, spacing):
    """(x,y,z) voxel indices -> the world coordinates centred_grid_3d uses."""
    nz, ny, nx = shape
    sz, sy, sx = spacing
    return (np.asarray(pts_xyz, float)
            - np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2])) * np.array([sx, sy, sz])

def cell_geometry(base, ttape, i, tg):
    """This cell's own geometry: the base shape plus its frozen size / orientation draws.
    """
    r = float(np.clip(base.RADIUS.v * np.exp(tg.SIZE_SIGMA.v * ttape["z_size"][i]),
                      base.RADIUS.lo, base.RADIUS.hi))
    nf = float(np.clip(base.NUC_FRAC.v * np.exp(tg.NUC_FRAC_SIGMA.v * ttape["z_nucfrac"][i]),
                       base.NUC_FRAC.lo, base.NUC_FRAC.hi))
    o = ttape["u_orient3"][i]
    return dataclasses.replace(base, RADIUS=r, NUC_FRAC=nf, POLAR_DEG=float(180.0 * o[0]),
                               AZIM_DEG=float(360.0 * o[1]), ROLL_DEG=float(360.0 * o[2]))

def thin(pts, order, min_dist):
    keep = np.ones(len(pts), bool)
    for i, j in cKDTree(pts).query_pairs(float(min_dist), output_type="ndarray"):
        keep[i if order[i] > order[j] else j] = False
    return np.flatnonzero(keep)


def support_mask(w, XY_scale_vox=40.0, Z_scale_vox=40.0, cover=0.75):
    """Smooth the frozen field, then threshold at a quantile so `cover` IS the realised fraction."""
    z = gaussian_filter(w.astype(np.float32),
                        (float(Z_scale_vox), float(XY_scale_vox), float(XY_scale_vox)), truncate=3.0)
    z = (z - z.mean()) / (z.std() + 1e-12)
    return z >= np.quantile(z, 1.0 - float(np.clip(cover, 0.0, 1.0)))


def assign_types(info, tape, Profiles, fractions, rule=None):
    """ecide each cell's type
    """
    type_names = list(Profiles)
    cuts = np.cumsum(np.asarray(fractions, float) / np.sum(fractions))
    types = {}
    for n in info["labels_present"]:
        i = info["cand_idx"][n]
        base = type_names[int(np.searchsorted(cuts, tape["u_type"][i]))]
        ctx = CellContext(label=n, index=i, centre=info["centres_vox"][n],
                          geom=info["geoms"][n], neigh_labels=info["neighbours"][n],
                          types=types)
        types[int(n)] = base if rule is None else rule(ctx, base)
    return types

class TapeDict(dict):
    """dict that also allows attribute access, so it works wherever a Tape does."""
    __getattr__ = dict.__getitem__

def build_tissue(tape, TG, shape, base_geom, spacing, Profiles, Fractions, um_per_vox, L = 4, l_min = 2):
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
    geoms, slices, centres_w = {}, {}, {}

    for n, i in enumerate(keep, start=1):
        g = cell_geometry(base_geom, tape, i, TG)
        sl = patch_grid(tape["xyz"][i],
                         reach_px(R=g.RADIUS.v, elong=g.ELONG.v, rough=g.ROUGH.v, grow=TG.GROW.v), 
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
        geoms[n], slices[n], centres_w[n] = g, sl, cw
        
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
            orphan += int(drop.sum())
    
    present = sorted(set(np.unique(labels)) - {0})
    cvox = tape["xyz"][keep]
    tree = cKDTree(cvox)
    neigh = {n: [j + 1 for j in tree.query_ball_point(cvox[n - 1], TG.NEIGH_RADIUS.v)
                 if j + 1 != n] for n in present}
    neigh = {n: [j for j in v if j in neigh] for n, v in neigh.items()}

    info = dict(labels_present=present, cand_idx={n: int(keep[n - 1]) for n in present},
        geoms=geoms, slices=slices, centres_w=centres_w,
        centres_vox={n: cvox[n - 1] for n in present},
        neighbours=neigh, support=sup, orphan_vox=orphan,
        packing=float((labels > 0).sum() / max(sup.sum(), 1)))
    
    types = assign_types(info, tape, Profiles, Fractions)
    
    names = sorted({m for p in Profiles.values() for m in p.Markers})
    out = {m: np.zeros(shape, np.float32) for m in names}
    grid = centred_grid_3d(shape, spacing)

    for n in info["labels_present"]:
        prof = Profiles[types[n]]
        g, sl = info["geoms"][n], info["slices"][n]
        win = labels[sl] == n
        if not win.any():
            continue
        # the packed body, so a marker fills the claimed territory, not just the free shape
        f = stamp_cell(tape, info["cand_idx"][n], g, grid, sl, info["centres_w"][n],
                          grow=TG.GROW.v)
        ptape = TapeDict(texture_noise=tape["texture_noise"][(slice(None),) + sl],
                         gate_noise=tape["gate_noise"][(slice(None),) + sl])
        off = 0
        for name, marker in prof.Markers.items():
            img = render_marker(tape = ptape, marker = marker, cell = f, spacing = spacing, geom = g, um_per_vox=um_per_vox,
                                edge_softness=0.0, pool_offset=off)
            out[name][sl] = np.where(win, img, out[name][sl])
            off += len(marker.noise_components)
    
    return out, labels, nuc_labels, tau_img, types, info

