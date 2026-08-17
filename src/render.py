from dataclasses import dataclass
import numpy as np


POOL_NAMES = ["diffuse", "fibrillar", "punctate"]
N_POOLS = 3


def localization_function(x, mu=0.0, width=0.5, sharp=4.0, floor=0.0):
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
    """Frozen noise through a transfer function, standardised to zero mean / unit variance."""
    z = np.fft.irfftn(np.fft.rfftn(noise) * H, s=noise.shape, axes=(0, 1, 2))
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


def pool_image(c, cell, tau, ctx):
    """One pool = its own localisation x its own texture, normalised to in-cell mean 1."""
    loc = localization_function(tau, mu=c.mu.v, width=c.width.v, sharp=c.sharp.v)
    f = loc * c.field_fct(ctx)
    return f / (f[cell].mean() + 1e-12)


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


def render_marker(
    tape, marker, cell, spacing, geom, um_per_vox=1.0, edge_softness=0.0, pool_offset=0
):
    """amp * normalise( sum_k w_k * normalise(loc_k x tex_k) x polarity x env(d) ).

    Normalise INSIDE each pool (a pool is a product), ADD across pools (means add), apply
    polarity, then multiply by the SMOOTH boundary envelope and rescale once so the marker's
    mean intensity over the interior AREA is exactly amp.
    """
    mask, tau, phi, d = cell["cell"], cell["tau"], cell["phi"], cell["d"]
    comps = marker.noise_components
    n_pool = tape["texture_noise"].shape[0]
    if pool_offset + len(comps) > n_pool:
        raise ValueError(
            f"marker {marker.name!r} needs pools {pool_offset}..{pool_offset + len(comps) - 1} "
            f"but the tape holds {n_pool}. The tape needs one pool per (marker, component) "
            f"pair -- call tape.draw3d(..., Pool=PROFILE.n_pools())."
        )

    f2, fpar2, fperp2 = _freq(mask.shape, spacing, geom.POLAR_DEG.v, geom.AZIM_DEG.v)

    out, wsum = np.zeros(mask.shape, float), 0.0
    for i, c in enumerate(comps):
        if c.w.v <= 0:
            continue
        j = pool_offset + i
        ctx = NoiseCtx(
            noise=tape["texture_noise"][j],
            gate=tape["gate_noise"][j],
            f2=f2,
            fpar2=fpar2,
            fperp2=fperp2,
            um_per_vox=um_per_vox,
        )
        out += c.w.v * pool_image(c, mask, tau, ctx)
        wsum += c.w.v
    out = out / wsum if wsum > 0 else mask.astype(float)

    if marker.polarity.v:
        pd = np.asarray([p.v for p in marker.pol_dir], np.float32)
        pd = pd / (np.linalg.norm(pd) + 1e-30)
        out = out * np.exp(np.float32(marker.polarity.v) * (phi @ pd))

    # hard by default: the soft rim in a real image is made by the optics, and psf_project
    # makes it. Anything but 0 here is blur invented before the microscope gets a look in.
    out = out * (boundary_falloff(d, edge_softness) if edge_softness > 0 else mask)

    denom = out.sum() / max(int(mask.sum()), 1)
    return (marker.amp.v * out / (denom + 1e-12)).astype(np.float32)


def render_image(tape, profile, cell, spacing, um_per_vox=1.0, edge_softness=0.0):
    """Render every marker of a CellProfile on one cell. Returns {marker name: volume}."""
    out, offset = {}, 0
    for name, m in profile.Markers.items():
        out[name] = render_marker(
            tape,
            m,
            cell,
            spacing,
            profile.Geometry,
            um_per_vox=um_per_vox,
            edge_softness=edge_softness,
            pool_offset=offset,
        )
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
    