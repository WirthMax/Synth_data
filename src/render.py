import numpy as np
from scipy.ndimage import gaussian_filter

POOL_NAMES = ["diffuse", "fibrillar", "punctate"]
N_POOLS = 3
_FIELD_KEYS = ("kind", "scale_px", "wavelength_px", "coherence", "length_px", "slope", "angle_deg")


def localization_function(x, mu=0.0, width=0.5, sharp=4.0, floor=0.0):
    f = np.exp(-np.abs((x - mu) / width) ** sharp)
    return (1.0 - floor) * f + floor

def build_comps(w0, s0, mu0, width0, sharp0, scale0,
                w1, s1, mu1, width1, sharp1, lam1, len1, ang1,
                w2, s2, mu2, width2, sharp2, scale2):
    return [
        dict(w=w0, strength=s0, mu=mu0, width=width0, sharp=sharp0,
             kind="blob", scale_px=scale0),
        dict(w=w1, strength=s1, mu=mu1, width=width1, sharp=sharp1,
             kind="stripe", wavelength_px=lam1, length_px=len1, angle_deg=ang1),
        dict(w=w2, strength=s2, mu=mu2, width=width2, sharp=sharp2,
             kind="blob", scale_px=scale2),
    ]

def _freq(shape, angle_deg=0.0):
    """Frequency grids (cycles/px) rotated into the texture's own frame."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    t = np.deg2rad(angle_deg)
    return fx*np.cos(t) + fy*np.sin(t), -fx*np.sin(t) + fy*np.cos(t)   # f_u (along), f_v
 
def field(noise, kind="blob", scale_px=4.0, wavelength_px=8.0, coherence=2.0,
          length_px=20.0, slope=1.5, angle_deg=0.0):
    fu, fv = _freq(noise.shape, angle_deg)
    f2 = fu**2 + fv**2
    if kind == "blob":
        H = np.exp(-2*np.pi**2 * scale_px**2 * f2)
    elif kind == "stripe":
        f0, sf = 1.0/wavelength_px, 1.0/(wavelength_px*coherence)
        H = np.exp(-(np.abs(fu) - f0)**2 / (2*sf**2)) * np.exp(-2*np.pi**2 * length_px**2 * fv**2)
    elif kind == "network":
        f0, sf = 1.0/wavelength_px, 1.0/(wavelength_px*coherence)
        H = np.exp(-(np.sqrt(f2) - f0)**2 / (2*sf**2))
    elif kind == "cloud":
        H = np.zeros_like(f2); nz = f2 > 0
        H[nz] = f2[nz]**(-slope/2)
        H *= np.exp(-2*np.pi**2 * scale_px**2 * f2)
    else:
        raise ValueError(kind)
    z = np.fft.ifft2(np.fft.fft2(noise) * H).real
    return z / (z.std() + 1e-12)

def boundary_falloff(d, edge_softness=0.06, edge_level=0.5):
    """Smooth cell-support envelope in the normalised radius d = rho / r(phi).
    Marker fades ACROSS the outline instead ofbeing clipped at it. 
    """
    p = float(np.clip(edge_level, 1e-3, 1.0 - 1e-3))
    s = max(float(edge_softness), 1e-6)
    c = 1.0 + s * np.log(p / (1.0 - p))          # env(1) == p
    return 1.0 / (1.0 + np.exp(np.clip((d - c) / s, -700.0, 700.0)))


def pool_image(c, cell, tau, tape_plane):
    """One pool = its own localisation x its own texture, normalised to in-cell mean 1."""
    loc = localization_function(tau, mu=c["mu"], width=c["width"], sharp=c["sharp"])
    tex = np.exp(c["strength"] * field(tape_plane, **{k: c[k] for k in _FIELD_KEYS if k in c}))
    f = loc * tex
    return f / (f[cell].mean() + 1e-12)


def render_marker(comps, cell, tau, phi, d, tape, polarity=0.0, pol_dir=0.0, amp=1.0,
                  edge_softness=0.06, edge_level=0.5):
    """amp * normalise( sum_k w_k * normalise(loc_k x tex_k) x polarity x env(d) ).

    Normalise INSIDE each pool (a pool is a product), ADD across pools (means add), apply
    polarity, then multiply by the SMOOTH boundary envelope and rescale once so the marker's
    mean intensity over the interior AREA is exactly amp. `d` is the normalised radius from
    cell_fields; edge_softness / edge_level tune the falloff (see boundary_falloff).
    """
    out, wsum = np.zeros(cell.shape, float), 0.0
    for i, c in enumerate(comps):
        if c["w"] <= 0:
            continue
        out += c["w"] * pool_image(c, cell, tau, tape["noise"][i])
        wsum += c["w"]
    out = out / wsum if wsum > 0 else cell.astype(float)
    if polarity:
        out = out * np.exp(polarity * np.cos(phi - pol_dir))
    out = out * boundary_falloff(d, edge_softness, edge_level)   # smooth support, not a hard cut
    # Normalise by the marker's total energy over the interior AREA -- not the interior MEAN.
    # The two agree for interior markers (skirt energy ~0), but when a marker peaks at or beyond
    # the boundary (mu >= 1) the interior is dark, its mean collapses to ~0, and dividing by it
    # would blow the boundary ring up without bound. Total-energy / area stays finite.
    denom = out.sum() / max(int(cell.sum()), 1)
    return amp * out / (denom + 1e-12)

def render_cell(cell, markers):
    hi  = max(np.percentile(im[cell], 99.5) for im in markers)
    return np.dstack([np.clip(im / hi + 1e-9, 0, 1) for im in markers])