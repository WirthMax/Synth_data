
import numpy as np
from scipy import ndimage as ndi


_NOISE_CAL = 0.85
IDX6 = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
_S2 = 1.0 / np.sqrt(2.0)
_S6 = 1.0 / np.sqrt(6.0)
E5 = np.array([
    [-_S6, -_S6, 2.0 * _S6, 0.0, 0.0, 0.0],
    [_S2, -_S2, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, _S2, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, _S2, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, _S2],
])

def unit(v, axis=0):
    """v / |v|, safe at zero."""
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12)

def to33(q6):
    """(6, ...) -> (..., 3, 3), the layout numpy.linalg.eigh wants."""
    q = np.asarray(q6, np.float64)
    A = np.empty(q.shape[1:] + (3, 3), np.float64)
    for k, (i, j) in enumerate(IDX6):
        A[..., i, j] = q[k]
        A[..., j, i] = q[k]
    return A

def frob(q6):
    """The Frobenius norm squared, sum_ij Q_ij^2."""
    q = np.asarray(q6)
    return q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + 2.0 * (q[3] ** 2 + q[4] ** 2 + q[5] ** 2)

# Q = w (n(x)n - I/3), stored as its 6 independent components (xx,yy,zz,xy,xz,yz)
def uni6(n, w):
    """A director as a Q-tensor, in 6-component storage: Q = w (n (x) n - I/3)"""
    n = unit(n)
    w = np.asarray(w, np.float64)
    return np.stack([w * (n[i] * n[j] - (1.0 / 3.0 if i == j else 0.0)) for i, j in IDX6])


def decompose(q6):
    """Q -> (director (3, ...), eigenvalue gap (...))."""
    # ascending
    lam, vec = np.linalg.eigh(to33(q6))          
    return np.moveaxis(vec[..., -1], -1, 0), lam[..., -1] - lam[..., -2]

def coherence(gap): 
    """gap -> S in [0, 1). """
    return gap / (1.0 + gap)


def noise6(c, w):
    """An ISOTROPIC random Q-tensor from five iid unit-variance fields, plus its own weight map."""
    c = np.asarray(c, np.float64)
    q = np.einsum("kc,k...->c...", E5, c)
    wn = float(w) * _NOISE_CAL
    return wn * q, wn * np.sqrt(1.5 * frob(q))

def slerp_axis(d, n, a):
    """Rotate the axis `d` a fraction `a` of the way toward the axis `n`, along the shortest arc.
    """
    d = unit(np.asarray(d, np.float64))
    n = unit(np.asarray(n, np.float64))
    a = float(a)
    if a <= 0.0:
        return d
    # the near end of the headless axis
    if float(d @ n) < 0.0:
        n = -n                                   
    g = float(np.arccos(np.clip(float(d @ n), -1.0, 1.0)))
    if g <= 1e-12:
        return d
    k = unit(np.cross(d, n))
    t = a * g
    return (d * np.cos(t) + np.cross(k, d) * np.sin(t) + k * float(k @ d) * (1.0 - np.cos(t)))

def sample_field(field, xs, ys, zs):
    """The field at many cell centres at once: the director from the interpolated Q, plus every
    named scalar feature the builder attached under field["features"].

    Note: a feature named "S" (the builder always supplies one) is returned by INTERPOLATING the
    coherence volume, not by decomposing the per-cell interpolated Q -- the two agree to within
    trilinear-interpolation error.
    """
    c = np.stack([np.asarray(zs, float), np.asarray(ys, float), np.asarray(xs, float)])
    q = np.stack([ndi.map_coordinates(field["Q6"][k], c, order=1, mode="nearest")
                  for k in range(6)])
    n, gap = decompose(q)
    m = np.stack([ndi.map_coordinates(mk, c, order=1, mode="nearest") for mk in field["m"]]) \
        if len(field["m"]) else np.zeros((0, c.shape[1]))
    out = {"n": n, "theta": np.arctan2(n[1], n[0]), "m": m}
    for name, vol in field.get("features", {}).items():
        out[name] = ndi.map_coordinates(vol, c, order=1, mode="nearest")
    out.setdefault("S", coherence(gap))          # fallback for a bare field with no features
    return out


def field_plane(field, z):
    """One lateral plane, decomposed into exactly the 2-D arrays the plot views expect.
    """
    z = int(np.clip(z, 0, field["Q6"].shape[1] - 1))
    n, gap = decompose(field["Q6"][:, z])
    th = np.arctan2(n[1], n[0])
    inp = np.hypot(n[0], n[1])
    n_st = len(field["dist"])
    comp = field.get("compliance")
    return {"n": n.astype(np.float32),
            "T": (inp * np.stack([np.cos(2 * th), np.sin(2 * th)])).astype(np.float32),
            "theta": th.astype(np.float32),
            "tilt": np.abs(n[2]).astype(np.float32),
            "S": coherence(gap).astype(np.float32),
            "features": {k: np.asarray(v)[z] for k, v in field.get("features", {}).items()},
            "resp": field["resp"][:, z],
            "dist": field["dist"][:, z] if n_st else np.zeros((0,) + th.shape, np.float32),
            "terr": field["terr"][:, z] if n_st else np.zeros((0,) + th.shape, bool),
            "lumen": field["lumen"][z] if "lumen" in field else np.zeros(th.shape, bool),
            "levels": field["levels"], "walls": field["walls"],
            "compliance": (comp[:, z] if comp is not None
                           else np.ones((n_st,) + th.shape, np.float32)),
            "contact": float(field.get("contact", 0.0)),
            "psi": field["psi"][z], "thr": field["thr"],
            "m": field["m"][:, z] if len(field["m"]) else np.zeros((0,) + th.shape, np.float32),
            "term_names": field["term_names"], "z": z}
