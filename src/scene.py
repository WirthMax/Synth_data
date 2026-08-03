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


def cell_fields(rho, phi, r_cell, r_nuc=None, rim=1.5):
    """Masks and intrinsic coordinates from the two support functions.

    d = rho / r_cell is exactly 1 on the free boundary, which is what the tessellation
    compares across cells. The sandbox just thresholds it at 1.
    """
    d = rho / np.maximum(r_cell, 1e-6)
    out = {"d": d, "cell": d <= 1.0, "phi": phi, "rho": rho}
    if r_nuc is not None:
        # Clip the NUCLEAR SUPPORT, then build both the mask and tau from that same clipped
        # curve. Deriving the mask from the clipped radius but tau from the unclipped one
        # desynchronises them: tau = 0 lands somewhere other than the labelled nuclear edge,
        # so a nuclear marker bleeds across the boundary of its own label.
        r_n = np.maximum(np.minimum(r_nuc, r_cell - rim), 1e-6)
        out["r_nuc_eff"] = r_n
        out["nuc"] = rho <= r_n
        out["tau"] = np.where(rho < r_n, rho / r_n - 1.0,
                              (rho - r_n) / np.maximum(r_cell - r_n, 1e-6))
    return out


def generate_single_cell(Tape, size=201, radius=32, nuc_frac=0.3, rough=0.25,
                         elong=1.6, angle_deg=30, beta=1.9, rim=1.5):
    """Sandbox: one cell centred in its own image."""
    # take the first candidate from the tape
    r_cell_fn = make_boundary(Tape.a[0], Tape.b[0], radius, rough, beta, Tape.K)
    r_nuc_fn = make_boundary(Tape.a[0], Tape.b[0], radius * nuc_frac, rough * 0.6, beta, Tape.K, extra=1.5)
    y, x = centred_grid(size)
    rho, phi = body_frame(y, x, 0.0, 0.0, angle_deg, elong)
    return cell_fields(rho, phi, r_cell_fn(phi), r_nuc_fn(phi), rim)

### Tissue
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

def thin(tape, min_dist):
    """Each candidate carries a frozen random priority order. A candidate is dropped if any higher-priority candidate lies within min_dist
    """
    xy, order = tape["xy"], tape["order"]
    tree = cKDTree(xy)
    keep = np.ones(len(xy), bool)
    # return pairs closer than min_dist, and drop the one with lower priority
    for i, j in tree.query_pairs(min_dist, output_type="ndarray"):
        keep[i if order[i] > order[j] else j] = False
    k = np.flatnonzero(keep)
    d = cKDTree(tape["xy"][k]).query(tape["xy"][k], k=2)[0][:, 1].min()
    assert d >= min_dist, f"min dist {d} < {min_dist}"
    return k

def support_mask(tape, scale_px=40.0, cover=0.75):
    """ generate tissue mask by using gaussian filter on the support field and thresholding it to get a binary mask
    """
    z = gaussian_filter(tape["support"], scale_px, truncate=4.0)
    z = (z - z.mean()) / (z.std() + 1e-12)
    return z >= np.quantile(z, 1.0 - np.clip(cover, 0.0, 1.0))


def stamp_cell(tape_a, tape_b, cy, cx, tile, radius, nuc_frac, rough, elong,
               angle_deg, beta, K, grow=1.0, rim=1.5):
    """Tissue: one cell at (cy, cx) on a local patch. Same three calls as in generate_single_cell."""
    r_cell_fn = make_boundary(tape_a, tape_b, radius, rough, beta, K)
    r_nuc_fn = make_boundary(tape_a, tape_b, radius * nuc_frac, rough * 0.6, beta, K, extra=1.5)
    y, x, origin = patch_grid(cy, cx, reach_px(radius, elong, rough, grow), tile)
    rho, phi = body_frame(y, x, cy, cx, angle_deg, elong)
    return cell_fields(rho, phi, r_cell_fn(phi), r_nuc_fn(phi), rim), origin