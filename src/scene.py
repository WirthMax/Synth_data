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
from scipy import ndimage as ndi



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


def build_tissue(tape, min_dist=11.0, radius=8.0, size_sigma=0.15, elong=1.6,
                 rough=0.15, beta=2.0, support_scale=40.0, cover=0.75, grow=1.35,
                 nuc_frac=0.35, nuc_frac_sigma=0.15, rim=1.5, nuc_corr=0.5, nuc_offset=0.9,):
    """Render a tile of packed cells: instance labels plus intrinsic coordinates.

    Nothing here reads pixel data. The output is a deterministic function of (tape, theta),
    which is what makes the label trustworthy as ground truth.

    Parameters
    ----------
    tape : dict
        Frozen randomness, drawn once and independent of every parameter below. Fields used
        here: xy (candidate centres), order (hard-core priority), a/b (cell harmonics),
        a2/b2 (independent nuclear harmonics), z_size, z_nucfrac, u_orient, u_offdir,
        u_offmag, support, tile, K. Parameters only THRESHOLD or SMOOTHLY MAP these, never
        resample them -- resampling per theta would make the fitting objective jagged.

    Point process
    -------------
    min_dist : float, px
        Hard-core radius: no two surviving centres are closer than this. GROUNDED -- measure
        it from real nearest-neighbour distances rather than fitting it, since it defines the
        mask. Roughly 1.2-1.8 x radius; too large and only a handful of candidates survive.
    support_scale : float, px
        Correlation length of the tissue-support field. Must stay >> cell size (>= ~4 x
        radius) so the support boundary can never be mistaken for a cell edge.
    cover : float in [0, 1]
        Fraction of the tile that is tissue. Applied as a quantile of the support field, so
        it maps monotonically onto realised coverage whatever the field's spread.

    Cell geometry (all mask-defining -> GROUNDED, not fitted)
    --------------------------------------------------------
    radius : float, px
        Median equivalent-circle radius of the FREE shape (area = pi r^2). Note the realised
        label area differs after packing -- ground this against post-packing median area,
        not against pi*radius^2.
    size_sigma : float
        Lognormal spread of cell size: realised r = radius * exp(size_sigma * z_size).
        0.15 gives roughly +-15%.
    elong : float >= 1
        Median aspect ratio. Area-preserving (the body-frame map has unit determinant), so
        this does not double as a size knob.
    rough : float
        Boundary irregularity, read as the SD of log radius: 0.15 ~ +-15% radial wobble.
        Above ~0.35 territories start pinching apart -- watch info["orphan_px"].
    beta : float
        Spectral tilt of the boundary harmonics at FIXED amplitude. Large -> a few fat
        lobes; small -> fine crenulation, which pinches at small radius. Below ~1.0 with
        radius < 5 px the shape is no longer band-limited for the pixel grid.

    Packing
    -------
    grow : float >= 1
        How far a cell may claim pixels, in units of its own free boundary (d = rho/r).
        1.0 -> free shapes with gaps between them; ~2 -> confluent, cells meeting along
        contact surfaces. In dense regions a neighbour binds first and grow is inert; in
        sparse regions grow alone sets the cell size.

    Nucleus
    -------
    nuc_frac : float
        Median nucleus:cell equivalent-radius ratio.
    nuc_frac_sigma : float
        Per-cell lognormal spread of that ratio. Set > 0, or nuclear area predicts cell area
        exactly and a nucleus-only model can invert the whole segmentation.
    nuc_corr : float in [0, 1]
        How much the nuclear outline mirrors the cell outline. 1 -> a scaled copy (leaks
        orientation and elongation); 0 -> independent. Implemented by mixing two frozen
        draws, so it stays smooth and safe to fit.
    nuc_offset : float in [0, 1]
        Nuclear eccentricity, as a fraction of the free cytoplasmic room. 0 puts the nucleus
        exactly on the tessellation seed, making the seed exactly recoverable from the
        nucleus.
    rim : float, px
        Minimum cytoplasmic clearance between the nuclear and plasma membranes. Enforced on
        the support function, so the labelled nuclear edge and the tau = 0 level set stay the
        same curve. Must stay resolvable -- below ~1 px it vanishes after the PSF.

    Returns
    -------
    labels : (tile, tile) int32       0 = background, k = cell k. THE GROUND TRUTH.
    nuc_labels : (tile, tile) int32   nuclei, same ids as `labels`.
    tau_img : (tile, tile) float      -1 nucleus centre, 0 nuclear envelope, +1 free
                                      membrane. Exceeds 1 in contact zones when grow > 1.
    phi_img : (tile, tile) float      body-frame angle, co-rotating with each cell.
    info : dict                       n_cells, centres, r_eff, support, orphan_px, empty,
                                      packing (cell pixels / support pixels).
    """
    tile = tape["tile"]
    # generate a support mask to limit the area where cells can be placed
    sup = support_mask(tape, support_scale, cover)
    # keep only the candidates that are sufficiently far apart and within the support mask
    keep = thin(tape, min_dist)
    cy0, cx0 = tape["xy"][keep].T
    keep = keep[sup[np.clip(cy0.astype(int), 0, tile-1),
                    np.clip(cx0.astype(int), 0, tile-1)]]

    best       = np.full((tile, tile), np.inf)
    labels     = np.zeros((tile, tile), np.int32)
    nuc_labels = np.zeros((tile, tile), np.int32)
    tau_img    = np.zeros((tile, tile))
    phi_img    = np.zeros((tile, tile))
    r_eff = radius * np.exp(size_sigma * tape["z_size"][keep])


    nf = np.clip(nuc_frac * np.exp(nuc_frac_sigma * tape["z_nucfrac"][keep]), 0.15, 0.85)

    for n, i in enumerate(keep, start=1):
        cy, cx = tape["xy"][i]
        
        f, (y0, x0) = stamp_cell(
            tape = tape, i = i, cy = cy, cx = cx, tile = tile, radius = r_eff[n-1], 
            nuc_frac = nf[n-1], rough = rough, elong = elong, 
            angle_deg = 180.0 * tape["u_orient"][i], beta = beta,
            grow=grow, rim=rim, nuc_corr=nuc_corr, nuc_offset=nuc_offset, 
        )
        h, w = f["d"].shape
        if h == 0 or w == 0:
            continue
        sl = (slice(y0, y0+h), slice(x0, x0+w))
        
        # tesselation to decide which cell is closest to each pixel 
        # -> find boundries
        win = (f["d"] < best[sl]) & (f["d"] <= grow) & sup[sl]
        best[sl] = np.where(win, f["d"], best[sl])
        labels[sl] = np.where(win, n, labels[sl])
        tau_img[sl] = np.where(win, f["tau"], tau_img[sl])
        phi_img[sl] = np.where(win, f["phi"], phi_img[sl])
        nuc_labels[sl] = np.where(win, f["nuc"]*n, nuc_labels[sl])
        
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

    present = np.flatnonzero(np.bincount(labels.ravel(), minlength=len(keep)+1)[1:])
    info = dict(n_cells=len(keep), centres=tape["xy"][keep], r_eff=r_eff, support=sup,
                orphan_px=orphan, empty=len(keep)-len(present),
                packing=float((labels > 0).sum() / max(sup.sum(), 1)))
    return labels, nuc_labels, tau_img, phi_img, info

