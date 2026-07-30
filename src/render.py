import numpy as np
from scipy.ndimage import gaussian_filter




def localization_function(x, mu=.0, width=0.5, sharp=4.0, floor=0.0):
    """ Where the marker is expressed as a function of tau, with a peak at mu, width, sharpness and floor level. """
    f = np.exp(-np.abs((x - mu) / width) ** sharp)
    return ((1.0 - floor) * f + floor)
     
     
def texture(shape, phi, tape, kind = 'blob', scale_px=4.0, wavelength_px=8.0, 
            coherence=2.0, length_px=20.0, slope=1.5, angle_deg=0.0, 
            strength=0.5, polarity=0.0, pol_dir=0.0, seed=0):
    """Blobby patchiness × angular polarity."""
    
    z = field(tape.noise, 
              kind = kind, 
              scale_px=scale_px,
              wavelength_px=wavelength_px, 
              coherence=coherence,
              length_px=length_px, 
              slope=slope, 
              angle_deg=angle_deg
              )
    z /= z.std() + 1e-12
    return np.exp(strength * z + polarity * np.cos(phi - pol_dir))

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


def render(cell, *factors, amp=1.0):
    """Multiply the shapes, gate by the mask, and set the amount -- normalise ONCE, last."""
    f = cell.astype(float)
    for x in factors:
        f = f * x
    return amp * f / f[cell].mean()