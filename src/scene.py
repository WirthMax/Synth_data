import numpy as np

def generate_single_cell(Tape, size = 201, radius = 32, nuc_frac = 0.3, rough = 0.9, K = 20, elong = 1.6, angle_deg = 30, beta = 1.9):
    # Random fourrier coefficients for a star shaped polygon
    a, b = Tape.a, Tape.b
    k = np.arange(1, K + 1)
    
    # get polar coordinates of the star shaped polygon
    # generate coordinate grid with origin in the center
    y, x = np.mgrid[:size, :size] - (size-1)/2
    t = np.deg2rad(angle_deg)
    # rotate the coordinate grid by angle_deg
    xr, yr = x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)
    # elongation factor along the x-axis
    s = np.sqrt(elong)
    # get polar coordinates
    rho = np.hypot(xr/s, yr*s)
    phi = np.arctan2(yr*s, xr/s)
    
    phi_q = np.linspace(0, 2*np.pi, 1024)          # dense angle grid, define once

    def boundary(R, kappa, extra = 0.0):
        # boundry radius as function of angle adding wobbles
        # get more fat lobes 
        w = kappa* k ** -(beta + extra)
        # decouple the scale from the amount of wobble
        w = w / (np.linalg.norm(w) + 1e-12) * kappa
        
        def g(p):
            # generate lumps at random locations (cosine and sine components)   
            return np.cos(p[..., None]*k) @ (w*a) + np.sin(p[..., None]*k) @ (w*b)
        
        area = 0.5 * np.trapezoid(np.exp(g(phi_q))**2, phi_q)   # area of THIS shape at R=1
        # rescale to exactly pi*R^2
        return R * np.sqrt(np.pi / area) * np.exp(g(phi))       
    

    # get boundaries
    r_cell = boundary(radius, rough)
    r_nuc  = boundary(radius * nuc_frac, rough * 0.6, extra=1.5)

    # check if inside cell / bucleus
    cell = rho <= r_cell
    nuc  = rho <= np.minimum(r_nuc, r_cell - 1.5)     # leave a cytoplasmic rim

    # depth: -1 nucleus centre, 0 envelope, +1 membrane
    tau = np.where(rho < r_nuc, rho/r_nuc - 1.0,
                   (rho - r_nuc) / np.maximum(r_cell - r_nuc, 1e-6))
    
    return cell, nuc, rho, phi, tau


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
