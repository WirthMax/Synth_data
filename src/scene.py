import numpy as np
from scipy.spatial.transform import Rotation as R


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
    w * (kappa * np.sqrt(4.0 * np.pi) / np.sqrt((w ** 2).sum() + 1e-30))
    c = w*coeff
    
    # Compute quadrate to pinpount size and then scale accordingly
    uq, wq = sphere_quadrature()
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

def body_frame(grid, rot, centre = (0.0, 0.0, 0.0), angle_deg=0.0, elong=1.0):
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


def mix_harmonics(sh, sh2, corr):
    """Nuclear coefficients correlated with the cell's at level `corr`, without resampling."""
    c = float(np.clip(corr, 0.0, 1.0))
    s = np.sqrt(1.0 - c * c)
    return c * sh + s * sh2

def to_world(offset_body, rot=None, elong=1.0):
    """Body-frame (un-stretched) offset -> world offset: re-apply the stretch, then rotate.

    A body-frame length means the same thing at every orientation only if it is built here
    and mapped out, never added directly in world coordinates.
    """
    o = np.asarray(offset_body, float) * _stretch(elong)
    return o if rot is None else np.asarray(rot, float) @ o

def nucleus_centre(centre, radius, nuc_frac, rim, elong, rot,
                   nuc_offset, off_dir, off_mag):
    """Compute the nucleus centre, given the cell centre and shape parameters.
    """
    # Compute freespace to membrane with rim space
    free = max(radius * (1.0 - nuc_frac) - rim, 0.0)
    
    off = np.asarray(off_dir, float)
    off = off / (np.linalg.norm(off) + 1e-30)
    return np.asarray(centre, float) + to_world(off * (nuc_offset * free * off_mag), rot, elong)


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
                      rough, beta, rot, angle_deg, elong, rim, nuc_offset, off_dir, off_mag, grow, 
                      centre = (0.0, 0.0, 0.0), rough_nuc = None, beta_nuc=None,
                      euclid_rim=True, nuc_fit_floor=None):
    """Shared core: both wrappers do exactly this, only the grid differs.

    body_grow scales the cell contour (not the nucleus); see cell_fields. Default 1.0 = the
    free shape used everywhere for the mask; > 1 gives the packed body for marker rendering.
    """
    cn = mix_harmonics(c_cell, c_nuc, nuc_corr)
    r_cell_fn = make_boundary(c_cell, radius, rough, beta, L, l_min)
    if rough_nuc is None:
        rough_nuc = rough * 0.6
    if rough_nuc is None:
        beta_nuc = beta
    r_nuc_fn = make_boundary(cn, radius * nuc_frac, rough_nuc, beta_nuc, L, l_min)
    rho_c, phi_c = body_frame(grid, rot = rot, centre = centre, angle_deg=angle_deg, elong=elong)
    print(rho_c.shape, phi_c.shape)
    ncentre = nucleus_centre(centre = centre, radius = radius, nuc_frac = nuc_frac, 
                             rim = rim, elong = elong, rot = rot, 
                             nuc_offset = nuc_offset, off_dir = off_dir, off_mag = off_mag)
    
    rho_n, phi_n = body_frame(grid, rot = rot, centre = ncentre, angle_deg=angle_deg, elong=elong)

    f = cell_fields(rho_c, phi_c, r_cell_fn(phi_c), 
                    rho_n, phi_n, r_nuc_fn(phi_n), 
                    rim, grow=grow)
    f["nuc_centre"] = ncentre
    f["r_cell_fn"], f["r_nuc_fn"] = r_cell_fn, r_nuc_fn
    return f


def generate_single_cell_3d(Tape, 
                            size=(128, 128, 128), spacing = 1.0, L=4, l_min=2,
                            i = 0, 
                            polar_deg=70.0, azim_deg=30.0, roll_deg=0.0,
                            nuc_corr=0.5, nuc_frac=0.25, radius=32, rough=0.25, rough_nuc =0.1, 
                            beta=0.5, beta_nuc=0.9, 
                            angle_deg = 0.0, elong = 1.0, rim = 1.5, nuc_offset = 0.6, 
                            grow = 1.0, euclid_rim=True, nuc_fit_floor=None):
    """Sandbox: candidate i from the tape, centred in its own image."""
    
    grid = centred_grid_3d(size, spacing=spacing)
    rot = rotation_matrix(polar_deg=polar_deg, azim_deg=azim_deg, roll_deg=roll_deg)
    return _cell_and_nucleus(
        c_cell = Tape["sh"][i], c_nuc = Tape["sh2"][i], grid = grid, L = L, l_min = l_min,
        nuc_corr = nuc_corr, nuc_frac = nuc_frac, radius = radius, rough = rough, rough_nuc =rough_nuc, 
        beta=beta, beta_nuc=beta_nuc,
        rot = rot, angle_deg = angle_deg, elong = elong, rim = rim, 
        nuc_offset = nuc_offset, off_dir=Tape["u_offdir"][i], off_mag=Tape["u_offmag"][i], grow = grow,
        euclid_rim=euclid_rim, nuc_fit_floor=nuc_fit_floor
        )
        

### Tissue
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi
from render import render_marker          # one-way: render.py imports no scene, so no cycle



def stamp_cell(tape, i, cy, cx, tile, radius, nuc_frac, rough, elong, angle_deg, beta,
               grow=1.0, rim=1.5, nuc_corr=0.4, nuc_offset=0.6, body_grow=1.0):
    """Tissue: candidate i stamped at (cy, cx) on a local patch.

    `grow` sizes the patch (reach); `body_grow` scales the rendered cell contour (see
    cell_fields). They are set together (body_grow=grow) to render markers on the packed body.
    """
    y, x, origin = patch_grid(cy, cx, reach_px(radius, elong, rough, grow), tile)
    f = _cell_and_nucleus(
        tape["a"][i], tape["b"][i], tape["a2"][i], tape["b2"][i], y, x, cy, cx,
        radius, nuc_frac, rough, beta, tape["K"], elong, angle_deg, rim,
        nuc_corr, nuc_offset,
        off_dir=2 * np.pi * tape["u_offdir"][i], off_mag=tape["u_offmag"][i],
        body_grow=body_grow)
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
                 nuc_frac=0.35, nuc_frac_sigma=0.15, rim=1.5, nuc_corr=0.5, nuc_offset=0.9,
                 markers=None):
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

    Markers (appearance -- opt-in; the mask is unchanged whether on or off)
    ----------------------------------------------------------------------
    markers : callable or None
        Each cell's channels are rendered on its own patch with
        `render_marker` and composited by the SAME tessellation winner as the label, so the
        image can never disagree with the mask. Texture comes from `tape["noise_tile"]` sliced
        to the patch, so it stays frozen and globally coherent across the tile.

    Returns
    -------
    labels : (tile, tile) int32       0 = background, k = cell k. THE GROUND TRUTH.
    nuc_labels : (tile, tile) int32   nuclei, same ids as `labels`.
    tau_img : (tile, tile) float      -1 nucleus centre, 0 nuclear envelope, +1 free
                                      membrane. Exceeds 1 in contact zones when grow > 1.
    phi_img : (tile, tile) float      body-frame angle, co-rotating with each cell.
    info : dict                       n_cells, centres, r_eff, support, orphan_px, empty,
                                      packing (cell pixels / support pixels), d_img (the winning
                                      normalised radius per pixel; inf in background), and
                                      markers (list of (tile, tile) channels, or None).
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
    channels   = None                    # allocated lazily once we know how many markers
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

        # markers: render this cell's channels and composite by the SAME winner mask, so the
        # image can never disagree with the label. Render on the GROWN body (body_grow=grow):
        # its boundary is grow * r(phi), which coincides with the label extent (f["d"] <= grow),
        # so a marker fills the packed territory instead of only the free shape. Same patch,
        # so the offsets/shape match the mask fields above.
        if markers is not None:
            fm, _ = stamp_cell(
                tape=tape, i=i, cy=cy, cx=cx, tile=tile, radius=r_eff[n-1],
                nuc_frac=nf[n-1], rough=rough, elong=elong,
                angle_deg=180.0 * tape["u_orient"][i], beta=beta,
                grow=grow, rim=rim, nuc_corr=nuc_corr, nuc_offset=nuc_offset, body_grow=grow)
            if fm["cell"].any():
                specs = markers()                    # [(comps, amp, polarity, pol_dir), ...]
                if channels is None:
                    channels = [np.zeros((tile, tile)) for _ in specs]
                ptape = {"noise": tape["noise_tile"][:, y0:y0+h, x0:x0+w]}   # render_marker reads only ["noise"]
                for ch, (comps, amp, pol, pdir) in zip(channels, specs):
                    img = render_marker(comps, fm["cell"], fm["tau"], fm["phi"], fm["d"],
                                        ptape, pol, pdir, amp)
                    ch[sl] = np.where(win, img, ch[sl])

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
            if channels is not None:
                for ch in channels:
                    ch[slc][drop] = 0.0          # keep the image matching the trimmed mask
            orphan += int(drop.sum())

    present = np.flatnonzero(np.bincount(labels.ravel(), minlength=len(keep)+1)[1:])
    info = dict(n_cells=len(keep), centres=tape["xy"][keep], r_eff=r_eff, support=sup,
                orphan_px=orphan, empty=len(keep)-len(present),
                packing=float((labels > 0).sum() / max(sup.sum(), 1)),
                d_img=best, markers=channels)
    return labels, nuc_labels, tau_img, phi_img, info

