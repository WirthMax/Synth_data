import numpy as np
from scipy.ndimage import gaussian_filter




def localization_function(x, mu=.0, width=0.5, sharp=4.0, floor=0.0):
    """ Where the marker is expressed as a function of tau, with a peak at mu, width, sharpness and floor level. """
    f = np.exp(-np.abs((x - mu) / width) ** sharp)
    return ((1.0 - floor) * f + floor)
     
     
def texture(shape, phi, strength=0.5, scale_px=3.0, polarity=0.0, pol_dir=0.0, seed=0):
    """Blobby patchiness × angular polarity."""
    rng = np.random.default_rng(seed)
    z = gaussian_filter(rng.standard_normal(shape), scale_px)
    z /= z.std() + 1e-12
    return np.exp(strength * z + polarity * np.cos(phi - pol_dir))


def render(cell, *factors, amp=1.0):
    """Multiply the shapes, gate by the mask, and set the amount -- normalise ONCE, last."""
    f = cell.astype(float)
    for x in factors:
        f = f * x
    return amp * f / f[cell].mean()