import numpy as np


def make_boundary(a, b, R, kappa, beta, K, extra=0.0, n_quad=1024):
    """Return r(phi): the support function of one contour, area normalised to pi*R^2.

    Returns a CALLABLE rather than an array, because the sandbox and the tile need it on
    different point sets. The expensive part (the area quadrature) happens once, here.
    """
    k = np.arange(1, K+1)
    w = kappa * k ** -(beta + extra)
    w = w / (np.linalg.norm(w) + 1e-12) * kappa
    wa, wb = w * a, w * b

    def g(p):
        return np.cos(p[..., None] * k) @ wa + np.sin(p[..., None] * k) @ wb

    phi_q = np.linspace(0.0, 2.0 * np.pi, n_quad)
    area = 0.5 * np.trapezoid(np.exp(g(phi_q)) ** 2, phi_q)   # area of this shape at R=1
    scale = R * np.sqrt(np.pi / area)
    return lambda p: scale * np.exp(g(p))


def body_frame(y, x, cy=0.0, cx=0.0, angle_deg=0.0, elong=1.0):
    """Image coordinates -> (rho, phi) in the cell's rotated, un-stretched frame."""
    t, s = np.deg2rad(angle_deg), np.sqrt(elong)
    dy, dx = y - cy, x - cx
    xr = dx * np.cos(t) + dy * np.sin(t)
    yr = -dx * np.sin(t) + dy * np.cos(t)
    return np.hypot(xr / s, yr * s), np.arctan2(yr * s, xr / s)


def centred_grid(size):
    """Full grid with the origin at the image centre -- the sandbox case."""
    y, x = np.mgrid[:size, :size] - (size - 1) / 2.0
    return y, x


def patch_grid(cy, cx, reach, tile):
    """Small grid around one centre, clipped to the tile -- the tissue case.

    Returns (y, x, (y0, x0)) so the caller can write results back at the right offset.
    """
    y0, y1 = max(0, int(cy) - reach), min(tile, int(cy) + reach + 1)
    x0, x1 = max(0, int(cx) - reach), min(tile, int(cx) + reach + 1)
    y, x = np.mgrid[y0:y1, x0:x1]
    return y.astype(float), x.astype(float), (y0, x0)


def reach_px(R, elong, rough, grow=1.0, pad=2):
    """Furthest pixel this cell can claim, in image px"""
    s = np.sqrt(elong)
    return int(np.ceil(R * grow * max(s, 1.0 / s) * np.exp(3.0 * rough))) + pad


def mix_harmonics(a, b, a2, b2, corr):
    """Nuclear coefficients correlated with the cell's at level `corr`, without resampling."""
    c = float(np.clip(corr, 0.0, 1.0))
    s = np.sqrt(1.0 - c * c)
    return c * a + s * a2, c * b + s * b2


def nucleus_centre(cy, cx, radius, nuc_frac, rim, elong, angle_deg,
                   nuc_offset, off_dir, off_mag):
    """Displace the nucleus inside its cell; returns the IMAGE-frame nuclear centre.

    `free` is a body-frame length, so the displacement is built in the body frame and then
    mapped back out: undo the stretch, then undo the rotation. Adding it directly in image
    coordinates would make nuc_offset mean something different at every orientation.
    """
    free = max(radius * (1.0 - nuc_frac) - rim, 0.0)
    m = nuc_offset * free * off_mag
    t, s = np.deg2rad(angle_deg), np.sqrt(elong)
    ox_b, oy_b = m * np.cos(off_dir), m * np.sin(off_dir)
    ox, oy = ox_b * s, oy_b / s
    return (cy + ox * np.sin(t) + oy * np.cos(t),
            cx + ox * np.cos(t) - oy * np.sin(t))


def cell_fields(rho_c, phi_c, r_c, rho_n=None, phi_n=None, r_n=None, rim=1.5):
    """Masks and coordinates from two supports about POSSIBLY DIFFERENT centres.

    psi = rho - r(phi) is signed (negative inside), measured relative to each contour --
    which is what lets the two be compared even though they have different origins.
    The containment clip is applied to psi_n ONCE and both outputs derive from it, so the
    labelled nuclear edge and the tau = 0 level set stay the same curve.
    """
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


def _cell_and_nucleus(a, b, a2, b2, y, x, cy, cx, radius, nuc_frac, rough, beta, K,
                      elong, angle_deg, rim, nuc_corr, nuc_offset, off_dir, off_mag):
    """Shared core: both wrappers do exactly this, only the grid differs."""
    an, bn = mix_harmonics(a, b, a2, b2, nuc_corr)          # nucleus ONLY
    r_cell_fn = make_boundary(a, b, radius, rough, beta, K)
    r_nuc_fn = make_boundary(an, bn, radius * nuc_frac, rough * 0.6, beta, K, extra=1.5)

    rho_c, phi_c = body_frame(y, x, cy, cx, angle_deg, elong)
    ncy, ncx = nucleus_centre(cy, cx, radius, nuc_frac, rim, elong, angle_deg,
                              nuc_offset, off_dir, off_mag)
    rho_n, phi_n = body_frame(y, x, ncy, ncx, angle_deg, elong)

    f = cell_fields(rho_c, phi_c, r_cell_fn(phi_c), rho_n, phi_n, r_nuc_fn(phi_n), rim)
    f["nuc_centre"] = (ncy, ncx)
    return f


def generate_single_cell(Tape, i=0, size=201, radius=32, nuc_frac=0.45, rough=0.25,
                         elong=1.6, angle_deg=30, beta=1.9, rim=1.5,
                         nuc_corr=0.4, nuc_offset=0.6):
    """Sandbox: candidate i from the tape, centred in its own image."""
    y, x = centred_grid(size)
    return _cell_and_nucleus(
        Tape["a"][i], Tape["b"][i], Tape["a2"][i], Tape["b2"][i], y, x, 0.0, 0.0,
        radius, nuc_frac, rough, beta, Tape["K"], elong, angle_deg, rim,
        nuc_corr, nuc_offset,
        off_dir=2 * np.pi * Tape["u_offdir"][i], off_mag=Tape["u_offmag"][i])


### Tissue
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter


def stamp_cell(tape, i, cy, cx, tile, radius, nuc_frac, rough, elong, angle_deg, beta,
               grow=1.0, rim=1.5, nuc_corr=0.4, nuc_offset=0.6):
    """Tissue: candidate i stamped at (cy, cx) on a local patch."""
    y, x, origin = patch_grid(cy, cx, reach_px(radius, elong, rough, grow), tile)
    f = _cell_and_nucleus(
        tape["a"][i], tape["b"][i], tape["a2"][i], tape["b2"][i], y, x, cy, cx,
        radius, nuc_frac, rough, beta, tape["K"], elong, angle_deg, rim,
        nuc_corr, nuc_offset,
        off_dir=2 * np.pi * tape["u_offdir"][i], off_mag=tape["u_offmag"][i])
    return f, origin


def thin(tape, min_dist):
    xy, order = tape["xy"], tape["order"]
    keep = np.ones(len(xy), bool)
    for i, j in cKDTree(xy).query_pairs(min_dist, output_type="ndarray"):
        keep[i if order[i] > order[j] else j] = False
    return np.flatnonzero(keep)


def support_mask(tape, scale_px=40.0, cover=0.75):
    z = gaussian_filter(tape["support"], scale_px, truncate=4.0)
    z = (z - z.mean()) / (z.std() + 1e-12)
    return z >= np.quantile(z, 1.0 - np.clip(cover, 0.0, 1.0))
