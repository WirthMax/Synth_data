from dataclasses import dataclass
import numpy as np
import scipy.fft as sfft

from render_numba import localization_numba


POOL_NAMES = ["diffuse", "fibrillar", "punctate"]
N_POOLS = 3

# scipy.fft thread count for the spectral noise-filtering step below -- -1 (all cores) by
# default, matching historic behaviour. A round-generation worker process that needs to stay
# single-threaded (OMP_NUM_THREADS=1 etc. alone does not bind scipy.fft, which takes its own
# explicit `workers` argument) sets `render.FFT_WORKERS = 1` once at process start.
FFT_WORKERS = -1


def localization_function(x, mu=0.0, width=0.5, sharp=4.0, floor=0.0):
    if localization_numba is not None:
        return localization_numba(x, mu, width, sharp, floor)
    f = np.exp(-(np.abs((x - mu) / width) ** sharp))
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
    f2 = fz**2 + fy**2 + fx**2
    dx, dy, dz = direction(polar_deg, azim_deg)
    fpar = fx * dx + fy * dy + fz * dz
    fpar2 = fpar**2
    return f2, fpar2, np.maximum(f2 - fpar2, 0.0)


def _filtered(noise, H):
    """Frozen noise through a transfer function, standardised to zero mean / unit variance.
    scipy's FFT thread count follows module-level `FFT_WORKERS` (-1, all cores, by default);
    these transforms are otherwise the render's dominant cost."""
    z = sfft.irfftn(sfft.rfftn(noise, workers=FFT_WORKERS) * H, s=noise.shape, axes=(0, 1, 2),
                    workers=FFT_WORKERS)
    z = z - z.mean()
    return (z / (z.std() + 1e-12)).astype(np.float32)


def boundary_falloff(d, edge_softness=0.06, edge_level=0.5):
    """Smooth cell-support envelope in the normalised radius d = rho / r(phi).
    Marker fades ACROSS the outline instead ofbeing clipped at it. 
    """
    p = float(np.clip(edge_level, 1e-3, 1.0 - 1e-3))
    s = max(float(edge_softness), 1e-6)
    c = 1.0 + s * np.log(p / (1.0 - p))  # env(1) == p
    return 1.0 / (1.0 + np.exp(np.clip((d - c) / s, -700.0, 700.0)))


def loc_norm(c, nuc_frac, radius_vox=None, n=96):
    """The brightness a pool's localisation band has WHERE THE MARKER IS, on an ideal whole cell:
    the band-weighted mean of the band, sum(loc^2 dV) / sum(loc dV), over a sphere of radius 1 with
    a concentric nucleus of radius `nuc_frac` (tau = r / nuc_frac - 1 inside the nucleus, the
    fraction of the way from envelope to membrane outside it; dV = 3 r^2 dr). ~1 for a flat-topped
    band, ~0.7 for a soft one. Dividing a pool by it makes a type's LEVEL the marker's mean
    brightness inside its own compartment.

    Two things this replaces, both wrong for tissue:
    - normalising by the mean over the voxels a cell happens to have in the block. A cap with no
      nucleus in it has loc ~ 5e-4 everywhere for a nuclear marker, and dividing by that in-patch
      mean painted the nuclear marker over the whole cap at full expression. Against the ideal
      cell a cut cell simply shows what its tau says it should -- there, almost nothing.
    - normalising to the mean over the WHOLE CELL. Then a compartment's brightness scaled with
      1 / its volume share: the same DAPI level was 2x brighter per nucleus voxel at
      NUC_FRAC 0.8 and 600x at 0.12, so big cells with small nuclei saturated the frame. A stain
      is a concentration; the level now says how bright the stained compartment is.

    `radius_vox` (the cell's SHORT semi-axis in voxels) quantises the distance to the membrane to
    whole voxels, as the voxelised territory does -- the outermost voxel layer is tau = 1 however
    thin the band, so even a hairline membrane band has somewhere to be."""
    r = (np.arange(n) + 0.5) / n
    nf = float(np.clip(nuc_frac, 1e-3, 1.0 - 1e-3))
    dn = np.maximum(r - nf, 0.0)
    de = 1.0 - r
    if radius_vox is not None and radius_vox > 0:
        de = np.floor(de * float(radius_vox)) / float(radius_vox)
    tau = np.where(r < nf, r / nf - 1.0, dn / np.maximum(dn + de, 1e-6))
    loc = localization_function(tau, mu=c.mu.v, width=c.width.v, sharp=c.sharp.v)
    w = 3.0 * r * r
    den = float((loc * w).sum())
    return float((loc * loc * w).sum() / den) if den > 1e-9 else 1.0


def pool_image(c, cell, tau, ctx, nuc_frac=None, radius_vox=None):
    """One pool = its own localisation x its own texture, normalised to in-cell mean 1.

    Sandbox (nuc_frac None): normalised to mean 1 over the cell's own voxels, exact when the
    whole cell is in the volume. Tissue (nuc_frac given): the band is normalised to mean ~1 INSIDE
    ITS OWN COMPARTMENT on an ideal whole cell (`loc_norm`) and the texture to its mean over the
    patch, so a cell cut anywhere shows the brightness its tau calls for -- see loc_norm."""
    loc = localization_function(tau, mu=c.mu.v, width=c.width.v, sharp=c.sharp.v)
    tex = c.field_fct(ctx)
    if nuc_frac is None:
        f = loc * tex
        return f / (f[cell].mean() + 1e-12)
    return loc * tex / (max(loc_norm(c, nuc_frac, radius_vox), 1e-3) * (float(tex.mean()) + 1e-12))


@dataclass
class NoiseCtx:
    """Everything a noise component needs, bundled so `field_fct` takes one argument."""

    noise: np.ndarray
    gate: np.ndarray
    f2: np.ndarray
    fpar2: np.ndarray
    fperp2: np.ndarray
    um_per_vox: float = 1.0

    def px(self, p):
        """A `P` holding microns -> lateral voxels. `spacing` already carries the anisotropy."""
        return max(float(p.v) / float(self.um_per_vox), 1e-6)


def _pool_field(bank_field, j, n_bank, shift_seed):
    """Field for logical pool `j` from a bank of `n_bank` stored fields.
    Keep a small bank and, when index j wraps, take the field ROLLED by a deterministic
    per-j offset. 
    """
    f = bank_field[j % n_bank]
    lap = j // n_bank
    if lap == 0:
        return f
    rng = np.random.default_rng(shift_seed + 7919 * j)
    sh = [int(rng.integers(0, s)) for s in f.shape]
    return np.roll(f, sh, axis=tuple(range(f.ndim)))


def render_marker(
    tape, marker, cell, spacing, geom, um_per_vox=1.0, edge_softness=0.0, pool_offset=0,
    pool_bank=None, shift_seed=None, nuc_frac=None, radius_vox=None
):
    """amp * normalise( sum_k w_k * normalise(loc_k x tex_k) x polarity x env(d) ).

    Normalise INSIDE each pool (a pool is a product), ADD across pools (means add), apply
    polarity, then multiply by the SMOOTH boundary envelope and rescale once so the marker's
    mean intensity over the interior AREA is exactly amp.

    `shift_seed=None` (the default) resolves to the tape's OWN seed, so which pool-bank fields
    get rolled at a wrap is diversified per frame like everything else on the tape, rather than
    shared across every frame via a fixed literal. Pass an explicit int only to reproduce old
    (pre-diversification) renders. When the tape itself has no integer seed (cfg.SEED=None, i.e.
    a fresh-entropy tape), there is nothing to diversify against, so fall back to 0.

    `nuc_frac` switches the normalisation from "mean over this mask is amp" (the sandbox, where the
    mask is the whole cell) to "mean inside the marker's own compartment is amp" (tissue, where the
    mask is the part of the cell's territory inside the block -- possibly a cap with no nucleus in
    it). See `loc_norm`. In that mode nothing is renormalised over the mask afterwards.
    """
    if shift_seed is None:
        shift_seed = int(tape.seed) if tape.seed is not None else 0
    mask, tau, phi, d = cell["cell"], cell["tau"], cell["phi"], cell["d"]
    comps = marker.noise_components
    n_pool = tape["texture_noise"].shape[0]
    n_bank = n_pool if not pool_bank else min(int(pool_bank), n_pool)
    if not pool_bank and pool_offset + len(comps) > n_pool:
        raise ValueError(
            f"marker {marker.name!r} needs pools {pool_offset}..{pool_offset + len(comps) - 1} "
            f"but the tape holds {n_pool}. Either draw the tape with "
            f"Pool=panel.n_pools(), or pass pool_bank=<n> to reuse fields at a shift."
        )
        
    f2, fpar2, fperp2 = _freq(mask.shape, spacing, geom.POLAR_DEG.v, geom.AZIM_DEG.v)

    out, wsum = np.zeros(mask.shape, float), 0.0
    for i, c in enumerate(comps):
        if c.w.v <= 0:
            continue
        j = pool_offset + i
        ctx = NoiseCtx(
            noise=_pool_field(tape["texture_noise"], j, n_bank, shift_seed),
            gate=_pool_field(tape["gate_noise"], j, n_bank, shift_seed + 104729),
            f2=f2,
            fpar2=fpar2,
            fperp2=fperp2,
            um_per_vox=um_per_vox,
        )
        out += c.w.v * pool_image(c, mask, tau, ctx, nuc_frac=nuc_frac, radius_vox=radius_vox)
        wsum += c.w.v
    out = out / wsum if wsum > 0 else mask.astype(float)

    if marker.polarity.v:
        pd = np.asarray([p.v for p in marker.pol_dir], np.float32)
        pd = pd / (np.linalg.norm(pd) + 1e-30)
        kap = float(marker.polarity.v)
        out = out * np.exp(np.float32(kap) * (phi @ pd))
        if nuc_frac is not None:
            # mean of exp(kappa cos) over the sphere is sinh(kappa) / kappa: divide it out, so
            # polarity moves the marker around the cell without making the cell brighter
            out = out / (np.sinh(abs(kap)) / max(abs(kap), 1e-6))

    # hard by default: the soft rim in a real image is made by the optics, and psf_project
    # makes it. Anything but 0 here is blur invented before the microscope gets a look in.
    out = out * (boundary_falloff(d, edge_softness) if edge_softness > 0 else mask)

    if nuc_frac is not None:
        return (marker.amp.v * out).astype(np.float32)
    denom = out.sum() / max(int(mask.sum()), 1)
    return (marker.amp.v * out / (denom + 1e-12)).astype(np.float32)

def matrix_volumes(tape, matrix, labels, support, edge_dist, spacing, um_per_vox):
    """The extracellular matrix of one tile, one volume per pattern in `matrix`
    (flexible_types.sample_background). A pattern is a texture -- fibrous, punctate, reticular --
    laid over everything inside the tissue support that no cell claimed (the interstitium), plus,
    if its `peri` is above 0, a band that decays over `peri_um` inward from every cell boundary:
    the basement membrane and the capillary bed that outline muscle cells in a real section. Its
    grain runs along the pattern's own in-plane `azim`. Each texture is normalised to mean 1 over
    the block, so a channel's matrix level is its mean brightness in open matrix.

    The textures come from tape.matrix_noise / matrix_gate (Tape.drawMatrix): frozen, like
    everything else; the parameters only shape and gate them."""
    shape = labels.shape
    gap = (support & (labels == 0)).astype(np.float32)
    cells = labels > 0
    out = []
    for j, mt in enumerate(matrix):
        f2, fpar2, fperp2 = _freq(shape, spacing, 90.0, float(mt["azim"]))
        n_f = tape["matrix_noise"].shape[0]
        ctx = NoiseCtx(noise=tape["matrix_noise"][j % n_f], gate=tape["matrix_gate"][j % n_f],
                       f2=f2, fpar2=fpar2, fperp2=fperp2, um_per_vox=um_per_vox)
        tex = np.asarray(mt["comp"].field_fct(ctx), np.float32)
        tex = tex / (float(tex.mean()) + 1e-12)
        occ = gap
        if mt["peri"] > 0.0:
            w = max(float(mt["peri_um"]) / float(um_per_vox), 1e-3)
            occ = occ + np.float32(mt["peri"]) * np.exp(-edge_dist / w) * cells
        out.append((tex * occ).astype(np.float32))
    return out


def render_image(tape, panel, cell, spacing, geom, um_per_vox=1.0, edge_softness=0.0,
                 expression=None, pool_offset=0, pool_bank=None, shift_seed=None):
    """Render every marker of a CellProfile on one cell. Returns {marker name: volume}."""
    out, offset = {}, pool_offset
    for name, m in panel.Markers.items():
        lvl = 1.0 if expression is None else float(expression.get(name, 0.0))
        # Only render markers that are expressed in the cell
        if lvl > 0.0:
            v = render_marker(
                tape, m, cell, spacing, geom,
                um_per_vox=um_per_vox, edge_softness=edge_softness,
                pool_offset=offset, pool_bank=pool_bank, shift_seed=shift_seed,
            )
            out[name] = v if lvl == 1.0 else (v * np.float32(lvl)).astype(np.float32)
        else:
            out[name] = np.zeros(cell["cell"].shape, np.float32)
        offset += len(m.noise_components)
    return out


def to_rgb(mask, channels, pct=99.5):
    """Stack marker volumes into an RGB volume, EACH channel scaled by ITS OWN percentile
    inside the mask. DISPLAY ONLY -- once per-marker gains are fittable this must not sit in
    the generation path, or it normalises away exactly the brightness differences being fitted.
    """
    return np.stack(
        [
            np.clip(im / (np.percentile(im[mask], pct) + 1e-9), 0, 1)
            for im in channels.values()
        ],
        -1,
    )
    