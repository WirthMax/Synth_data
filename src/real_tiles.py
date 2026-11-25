"""Real staged MACSIMA slides -> model-ready tiles at the training pixel size.

The inversion model is trained on tiles rendered at `cfg.UM_PER_VOX` (0.325 um/px) by a generator
whose geometry is in physical units, so a real tile is read at the SAME physical pixel size: a
window of `T x um_per_px / um_raw` native pixels, AREA-averaged down to T x T and re-quantised to
uint16. Area averaging is what a camera with bigger pixels would have recorded; the earlier notebook
path (`ndi.zoom(order=1)` over a 6.5x reduction) point-sampled instead, which aliased the texture,
left the per-pixel noise un-averaged and filled the backsub zero floor with interpolated values --
input statistics no synthetic tile ever has. That path survives as `mode="legacy"` only so its
numbers can be reproduced.

Channel names come from MCMICRO backsub's `markers_bs.csv` beside the image (its rows ARE the output
channels). A slide can carry several DAPI-filter rows (one per re-stain); exactly ONE of them goes to
the model as the nuclear counterstain -- the sharpest one -- because every synthetic panel has exactly
one. Display and pseudo-labels read the same channel.

Reading only: nothing here writes to the slide directories.
"""
import csv
import os
import re

import numpy as np
import tifffile
import zarr
from scipy import ndimage as ndi

import config as cfg

T_TILE = int(cfg.TISSUE_VOL[1])          # the training tile edge, px


def open_slide(path):
    """A staged backsub OME-TIFF as a lazily-read slide: the zarr pyramid, the native pixel size from
    the OME header, and the channel table from the `markers_bs.csv` written beside it. Returns a dict
    (path, levels, um_raw, rows, names, dapi_rows)."""
    tf = tifffile.TiffFile(path)
    m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tf.ome_metadata or "")
    if m is None:
        raise ValueError(f"{path} has no PhysicalSizeX in its OME header -- pass a staged backsub "
                         f"OME-TIFF (MCMICRO output), not a raw or re-saved image")
    csv_path = os.path.join(os.path.dirname(path), "markers_bs.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"no markers_bs.csv beside {path} -- the channel names come from the "
                                f"backsub step's csv; point at the image inside its background/ dir")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    z = zarr.open(tifffile.imread(path, aszarr=True), mode="r")
    if hasattr(z, "keys"):
        keys = sorted(z.keys(), key=int)
        levels = [z[k] for k in keys]
    else:
        levels = [z]
    if levels[0].shape[0] != len(rows):
        raise ValueError(f"{path} has {levels[0].shape[0]} channels but {csv_path} lists {len(rows)} "
                         f"-- the csv must be the markers_bs.csv of THIS backsub run")
    names = [r["marker_name"] for r in rows]
    dapi_rows = [i for i, r in enumerate(rows) if r.get("Filter", "").strip().upper() == "DAPI"]
    return dict(path=path, levels=levels, um_raw=float(m.group(1)), rows=rows, names=names,
                dapi_rows=dapi_rows or [0])


def _otsu_fg(d, sigma=1.0):
    from skimage.filters import threshold_otsu
    g = ndi.gaussian_filter(np.asarray(d, np.float32), sigma)
    return g > threshold_otsu(g)


def nuclear_contrast(d):
    """off/on-nucleus contrast of a nuclear-stain image: median outside the Otsu foreground over
    median inside. A real counterstain sits at 0.03-0.08 (Slide1-5 DAPI_cycle5) to ~0.35 (cycle-1
    DAPI on Slide6-10); a cycle-1 row that is mostly blue autofluorescence reads 0.6-0.7."""
    fg = _otsu_fg(d)
    if fg.all() or not fg.any():
        return 1.0
    return float(np.median(d[~fg]) / (np.median(d[fg]) + 1e-6))


def nuclear_channel(slide, positions, probe=600):
    """The DAPI-filter row with the sharpest nuclei (lowest off/on contrast, median over native
    `probe`-px windows at `positions`). With one DAPI row it is that row."""
    rows = slide["dapi_rows"]
    if len(rows) == 1:
        return rows[0]
    arr = slide["levels"][0]
    con = {i: float(np.median([nuclear_contrast(np.asarray(arr[i, r0:r0 + probe, c0:c0 + probe]))
                               for r0, c0 in positions])) for i in rows}
    return min(con, key=con.get)


def area_matrix(S, T):
    """[T, S] weights that area-average S native pixels onto T output pixels: output pixel i covers
    [i S/T, (i+1) S/T) and takes each native pixel with the length of their overlap, normalised to
    sum 1. Exact for non-integer ratios (306 -> 160 is 1.9125)."""
    f = S / T
    lo = np.arange(T)[:, None] * f
    hi = lo + f
    j = np.arange(S)[None, :]
    w = np.clip(np.minimum(hi, j + 1) - np.maximum(lo, j), 0.0, None)
    return (w / w.sum(1, keepdims=True)).astype(np.float32)


def model_channels(slide, nuclear):
    """Rows that go to the model: every antibody channel plus the ONE chosen nuclear row; the other
    DAPI-filter rows are dropped because a synthetic panel never has a second counterstain."""
    return [i for i in range(len(slide["names"])) if i == nuclear or i not in slide["dapi_rows"]]


def read_tile(slide, r0, c0, nuclear, um_per_px=None, T=T_TILE, mode="area", channels=None):
    """One real tile at the training pixel size. (r0, c0) is the native top-left corner; the native
    window is S = round(T x um_per_px / um_raw) px. `mode="area"` area-averages and rounds to uint16
    (the default, matched to training); `mode="legacy"` is the old notebook path (bilinear point
    sampling, float, no rounding). Returns dict(image [T, T, M] , names, nuclear_idx, S, um_per_px,
    rows)."""
    um_per_px = float(cfg.UM_PER_VOX if um_per_px is None else um_per_px)
    S = int(round(T * um_per_px / slide["um_raw"]))
    rows = model_channels(slide, nuclear) if channels is None else list(channels)
    arr = slide["levels"][0]
    raw = np.stack([np.asarray(arr[i, r0:r0 + S, c0:c0 + S]) for i in rows]).astype(np.float32)
    if raw.shape[1:] != (S, S):
        raise ValueError(f"window ({r0}, {c0}) + {S} px runs off the slide {arr.shape[1:]} -- pick a "
                         f"corner at least {S} px from the bottom/right edge")
    if mode == "area":
        W = area_matrix(S, T)
        out = np.einsum("ts,msr,ur->tum", W, raw, W, optimize=True)
        img = np.clip(np.rint(out), 0, 65535).astype(np.uint16)
    elif mode == "legacy":
        img = np.moveaxis(ndi.zoom(raw, (1, T / S, T / S), order=1)[:, :T, :T], 0, -1)
    else:
        raise ValueError(f"mode {mode!r} -- use 'area' (training-matched) or 'legacy' (old notebook)")
    return dict(image=img, names=[slide["names"][i] for i in rows], rows=rows,
                nuclear_idx=rows.index(nuclear) if nuclear in rows else -1,
                S=S, um_per_px=S * slide["um_raw"] / T)


def tissue_positions(slide, n, um_per_px=None, T=T_TILE, seed=0, min_tissue=0.4, level=None,
                     ref_channel=None):
    """`n` native top-left corners of non-overlapping tiles that lie on tissue, drawn reproducibly
    from `seed`. Tissue is found on a coarse pyramid level from the FIRST DAPI-filter row, which is
    either a stain or blue autofluorescence and in both cases lights every bit of tissue; a tile
    qualifies if at least `min_tissue` of its window is tissue. Returns a list of (r0, c0, frac)."""
    um_per_px = float(cfg.UM_PER_VOX if um_per_px is None else um_per_px)
    S = int(round(T * um_per_px / slide["um_raw"]))
    levels = slide["levels"]
    if level is None:                       # the coarsest level where a tile is still >= 8 px
        level = max(k for k in range(len(levels)) if S / 2 ** k >= 8)
    ch = slide["dapi_rows"][0] if ref_channel is None else ref_channel
    low = np.log1p(np.asarray(levels[level][ch]).astype(np.float32))
    low = ndi.gaussian_filter(low, 1.0)
    from skimage.filters import threshold_otsu
    tissue = low > threshold_otsu(low[low > 0]) if (low > 0).any() else np.zeros_like(low, bool)
    s = S / 2 ** level
    k = max(int(round(s)), 1)
    frac = ndi.uniform_filter(tissue.astype(np.float32), size=k, mode="constant")
    H0, W0 = levels[0].shape[1:]
    rng = np.random.default_rng(seed)
    cy, cx = np.nonzero(frac >= min_tissue)
    order = rng.permutation(cy.size)
    taken, out = [], []
    for i in order:
        r0 = int(round((cy[i] - k / 2) * 2 ** level))
        c0 = int(round((cx[i] - k / 2) * 2 ** level))
        if r0 < 0 or c0 < 0 or r0 + S > H0 or c0 + S > W0:
            continue
        if any(abs(r0 - a) < S and abs(c0 - b) < S for a, b in taken):
            continue
        taken.append((r0, c0))
        out.append((r0, c0, float(frac[cy[i], cx[i]])))
        if len(out) >= n:
            break
    return out


def find_slides(root="/mnt/Transfer/Myocarditis26"):
    """Every staged backsub OME-TIFF under `root`, sorted."""
    import glob
    return sorted(glob.glob(os.path.join(root, "*", "*", "*", "background", "*_backsub.ome.tif")))
