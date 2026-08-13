import numpy as np
from scipy.special import ndtri 

POOL_NAMES = ["diffuse", "fibrillar", "punctate"]
N_POOLS = 3
KINDS = ("blob", "cluster", "network", "fibre", "sheet")

def localization_function(x, mu=0.0, width=0.5, sharp=4.0, floor=0.0):
    f = np.exp(-np.abs((x - mu) / width) ** sharp)
    return (1.0 - floor) * f + floor

def direction(polar_deg=0.0, azim_deg=0.0):
    """Unit vector (x, y, z) from a polar angle off +z and an azimuth in the xy plane."""
    t, p = np.deg2rad(polar_deg), np.deg2rad(azim_deg)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])

def _freq(shape, spacing=1.0, polar_deg=0.0, azim_deg=0.0):
    """Frequency grids (cycles/px) rotated into the texture's own frame."""
    nz, ny, nx = shape
    sz, sy, sx = (spacing,) * 3 if np.isscalar(spacing) else spacing
    # Discrete Fourier Transform sample frequencies
    fz = np.fft.fftfreq(nz, sz)[:, None, None]
    fy = np.fft.fftfreq(ny, sy)[None, :, None]
    fx = np.fft.rfftfreq(nx, sx)[None, None, :]
    f2 = fz ** 2 + fy ** 2 + fx ** 2
    dx, dy, dz = direction(polar_deg, azim_deg)
    fpar = fx * dx + fy * dy + fz * dz
    fpar2 = fpar ** 2
    return f2, fpar2, np.maximum(f2 - fpar2, 0.0)
 
# def field(noise, c, spacing, polar_deg, azim_deg, slope=1.5, length_px=20.0
#          #   kind="blob", scale_px=4.0, wavelength_px=8.0, coherence=2.0,
#          #   length_px=20.0, slope=1.5, angle_deg=0.0
#            ):
#      f2, fpar2, fperp2 = _freq(shape = noise.shape, spacing = spacing, polar_deg = polar_deg, azim_deg = azim_deg)
#      if c['scope'] == "blob":
#          H = np.exp(-2 * np.pi ** 2 * c['scale'].v ** 2 * f2)
#      elif c['scope'] == "cloud":
#          H = np.zeros_like(f2); nz_ = f2 > 0
#          H[nz_] = f2[nz_] ** (-slope / 2)
#          H *= np.exp(-2 * np.pi ** 2 * c['scale'].v ** 2 * f2)
#      elif c['scope'] == "network":
#          f0, sf = 1.0 / c['scale'].v, 1.0 / (c['scale'].v * c['coherence'].v)
#          H = np.exp(-(np.sqrt(f2) - f0) ** 2 / (2 * sf ** 2))
#      elif c['scope'] == "fibre" or c['scope'] == "stripe":
#          H = np.exp(-2 * np.pi ** 2 * (c['lam'].v ** 2 * fperp2 + length_px ** 2 * fpar2))
#      elif c['scope'] == "sheet":
#          f0, sf = 1.0 / c['lam'].v, 1.0 / (c['lam'].v * c['coherence'].v)
#          H = (np.exp(-(np.sqrt(fpar2) - f0) ** 2 / (2 * sf ** 2))
#               * np.exp(-2 * np.pi ** 2 * length_px ** 2 * fperp2))
#      else:
#          raise ValueError(f"unknown kind {c['scope']}; expected one of {KINDS}")
#      z = np.fft.irfftn(np.fft.rfftn(noise) * H, s=noise.shape, axes=(0, 1, 2))
#      return (z / (z.std() + 1e-12)).astype(np.float32)


def _filtered(noise, H):
    """Frozen noise through a transfer function, standardised to zero mean / unit variance."""
    z = np.fft.irfftn(np.fft.rfftn(noise) * H, s=noise.shape, axes=(0, 1, 2))
    z = z - z.mean()
    return (z / (z.std() + 1e-12)).astype(np.float32)

def field(noise, c, spacing, polar_deg, azim_deg, um_per_vox=1.0, noise2=None):
    """The pool's TEXTURE -- non-negative, ready to multiply by the localisation.
    units are in MICRONS
    """
    f2, fpar2, fperp2 = _freq(shape=noise.shape, spacing=spacing,
                              polar_deg=polar_deg, azim_deg=azim_deg)
    scope = str(c['scope'])
    s = float(c['s'].v)
    # microns -> LATERAL VOXELS; `spacing` already carries the axial anisotropy
    px = lambda k: max(float(c[k].v) / float(um_per_vox), 1e-6)

    if scope == "blob":
        return np.exp(s * _filtered(noise, np.exp(-2 * np.pi ** 2 * px('scale') ** 2 * f2)))

    if scope == "cluster":
        # clusters of smaller blobs: fine speckle whose DENSITY is gated by a coarser field.
        # One field at one scale can only be uniformly grainy or uniformly lumpy; this is
        # patchy AND fine, which is what real vesicular staining looks like.
        fine = _filtered(noise, np.exp(-2 * np.pi ** 2 * px('scale') ** 2 * f2))
        coarse = _filtered(noise if noise2 is None else noise2,
                           np.exp(-2 * np.pi ** 2 * px('clust') ** 2 * f2))
        fill = float(np.clip(c['fill'].v, 1e-3, 1 - 1e-3))
        soft = max(float(c['soft'].v), 1e-3)
        # ndtri(1-fill) is the exact Gaussian quantile, so `fill` IS the occupied fraction
        gate = 1.0 / (1.0 + np.exp(-(coarse - ndtri(1.0 - fill)) / soft))
        return (gate * np.exp(s * fine)).astype(np.float32)

    if scope == "network":
        f0 = 1.0 / px('scale')
        sf = f0 / max(float(c['coherence'].v), 1e-3)
        return np.exp(s * _filtered(noise, np.exp(-(np.sqrt(f2) - f0) ** 2 / (2 * sf ** 2))))

    if scope in ("fibre", "stripe"):
        # anisotropic low-pass: thin across the axis (`lam`), long along it (`len`)
        H = np.exp(-2 * np.pi ** 2 * (px('lam') ** 2 * fperp2 + px('len') ** 2 * fpar2))
        return np.exp(s * _filtered(noise, H))

    if scope == "sheet":
        # band-pass ALONG the axis -> quasi-periodic bands (striations), coherent across it
        f0 = 1.0 / px('lam')
        sf = f0 / max(float(c['coherence'].v), 1e-3)
        H = (np.exp(-(np.sqrt(fpar2) - f0) ** 2 / (2 * sf ** 2))
             * np.exp(-2 * np.pi ** 2 * px('len') ** 2 * fperp2))
        return np.exp(s * _filtered(noise, H))

    raise ValueError(f"unknown kind {scope!r}; expected one of {KINDS}")

def boundary_falloff(d, edge_softness=0.06, edge_level=0.5):
    """Smooth cell-support envelope in the normalised radius d = rho / r(phi).
    Marker fades ACROSS the outline instead ofbeing clipped at it. 
    """
    p = float(np.clip(edge_level, 1e-3, 1.0 - 1e-3))
    s = max(float(edge_softness), 1e-6)
    c = 1.0 + s * np.log(p / (1.0 - p))          # env(1) == p
    return 1.0 / (1.0 + np.exp(np.clip((d - c) / s, -700.0, 700.0)))


def pool_image(c, cell, tau, noise, spacing, polar_deg, azim_deg, um_per_vox, noise2):
    """One pool = its own localisation x its own texture, normalised to in-cell mean 1."""
    loc = localization_function(tau, mu=c["mu"].v, width=c["width"].v, sharp=c["sharp"].v)
    f = loc * field(noise=noise, c=c, spacing=spacing, polar_deg=polar_deg,
                    azim_deg=azim_deg, um_per_vox=um_per_vox, noise2=noise2)
    return f / (f[cell].mean() + 1e-12)


def render_marker(tape, comps, cell, tau, phi, d, spacing, polar_deg,
    azim_deg, polarity=0.0, pol_dir=0.0, amp=1.0,
                  edge_softness=0.0, edge_level=0.5, um_per_vox=1.0):
    """amp * normalise( sum_k w_k * normalise(loc_k x tex_k) x polarity x env(d) ).

    Normalise INSIDE each pool (a pool is a product), ADD across pools (means add), apply
    polarity, then multiply by the SMOOTH boundary envelope and rescale once so the marker's
    mean intensity over the interior AREA is exactly amp. `d` is the normalised radius from
    cell_fields; edge_softness / edge_level tune the falloff (see boundary_falloff).
    """
    
    out, wsum = np.zeros(cell.shape, float), 0.0
    for i, c in enumerate(comps):
        if c["w"].v <= 0:
            continue
        out += c["w"].v * pool_image(c = c, cell = cell, tau = tau, noise = tape["texture_noise"][i], 
                                     spacing = spacing, polar_deg = polar_deg, azim_deg = azim_deg,
                                     um_per_vox = um_per_vox,
                                     noise2=tape["gate_noise"][i])
        wsum += c["w"].v
    out = out / wsum if wsum > 0 else cell.astype(float)
    if polarity:
        pd = np.asarray(pol_dir, np.float32)
        pd = pd / (np.linalg.norm(pd) + 1e-30)
        out = out * np.exp(np.float32(polarity) * (phi @ pd))
    out = out * boundary_falloff(d, edge_softness, edge_level) 
    
    denom = out.sum() / max(int(cell.sum()), 1)
    return (amp * out / (denom + 1e-12)).astype(np.float32)

def render_image(tape, p_dict, cell_mask, tau, phi, d, spacing, polar_deg, azim_deg, um_per_vox = 1.0):
    # Build kwargs for the underlying components (excluding the top-level rendering params)
    general = p_dict.pop('general', None)
    for marker, vals in p_dict.items():
        p_dict[marker] = render_marker(tape = tape, comps = vals, cell = cell_mask, tau = tau, phi = phi, d = d, spacing = spacing,
                         polar_deg = polar_deg, azim_deg = azim_deg, edge_softness=0.0, edge_level=0.5, amp = general['amp'].v, 
                         pol_dir = [x.v for x in general['pol_dir']], polarity = general['polarity'].v,
                         um_per_vox=um_per_vox)
            
    return p_dict, general


def to_rgb(mask, channels, pct=99.5):
    """Stack marker volumes into an RGB volume, EACH channel scaled by ITS OWN percentile
    inside the mask. DISPLAY ONLY -- once per-marker gains are fittable this must not sit in
    the generation path, or it normalises away exactly the brightness differences being fitted.
    """
    return np.stack(
        [np.clip(im / (np.percentile(im[mask], pct) + 1e-9), 0, 1) for im in channels.values()],
        -1,
    )