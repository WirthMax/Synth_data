import argparse, copy, time, os, re
import numpy as np

from scene import generate_single_cell_3d, build_tissue
from render import render_image, to_rgb
from tape import Tape
from plot import plot_surface_xyz_inline, plot_surface_xyz_html, plot_surface_rgb_html, ortho, ortho_rgb, tau_cmap, NormalizeData
from optics import kryostat, psf_project, mask_collapse, detector
from parameter import (P, CellGeometry, MarkerPanel, PanelMarker, CellType, Detector,
                       BlobNoise, ClusterNoise, NetworkNoise, FibreNoise, SheetNoise,
                       Optics, TissueGeometry, dapi_marker, DEVICE_DETECTOR, 
                       NOISE_KINDS, 
                       fitted_paths, set_vector, sample_prior, bounds)
import config as cfg


UM_PER_VOX = cfg.UM_PER_VOX
Z_RATIO = cfg.Z_RATIO
SPACING = cfg.SPACING
L, L_MIN = cfg.L, cfg.L_MIN
N_CAND, SIZE, TILE, K = cfg.N_CAND, cfg.SIZE, cfg.TILE, cfg.K
CELL_VOL = cfg.CELL_VOL
TISSUE_VOL = cfg.TISSUE_VOL
POOL_BANK = cfg.POOL_BANK
PIN_DEFAULT = cfg.PIN            # degenerate + grounded; see config
KEEP_DEFAULT = cfg.KEEP
DYES = cfg.DYES 
DAPI_NAME = cfg.DAPI_NAME
DAPI_DYE = cfg.DAPI_DYE

_W = {}
def _init_worker(seed, keep, cfg):
    _W["cfg"] = cfg
    _W["keep"] = keep
    _W["seed"] = seed
    
    

def panel_structure(n_markers, n_types, expr_frac, comps=(1, 4), struct_seed=0):
    """Which dye and which texture kinds each marker has, and
    which markers each cell type is positive for.
    (This is drawn once per dataset and then FIXED)
    """
    r = np.random.default_rng(struct_seed)
    # DAPI is always the first marker
    markers = [dict(name=DAPI_NAME, dye=DAPI_DYE, kinds=None)]
    for i in range(max(0, n_markers - 1)):
        k = int(r.integers(comps[0], comps[1] + 1))
        markers.append(dict(name=f"M{i:02d}", dye=str(r.choice(DYES)),
                            kinds=[int(x) for x in r.integers(0, len(NOISE_KINDS), size=k)]))
    ab = [m["name"] for m in markers if m["name"] != DAPI_NAME]
    support = []
    for t in range(n_types):
        on = ab if expr_frac >= 1.0 else [n for n in ab if r.random() < expr_frac]
        # every cell has a nucleus, so every type is DAPI-positive.
        support.append([DAPI_NAME] + (on or ([str(r.choice(ab))] if ab else [])))
    return dict(markers=markers, support=support)

def build_panel(struct, rng):
    """A MarkerPanel with the given structure and freshly drawn values."""
    markers = {}
    for spec in struct["markers"]:
        m = (dapi_marker() if spec["kinds"] is None else
             PanelMarker(name=spec["name"], fluorophore=spec["dye"],
                         noise_components=[NOISE_KINDS[k]() for k in spec["kinds"]]))
        # sampling respects each parameter's own bounds, so DAPI is drawn from its NARROWED
        # ranges and stays nuclear without any special case here
        paths = fitted_paths(m)
        set_vector(m, paths, sample_prior(rng, paths))
        markers[spec["name"]] = m
    return MarkerPanel(Markers=markers)

def build_cell_types(struct, rng, lo=0.3, hi=1.6, geom=None, name_fmt="T{:03d}"):
    """Cell types over the fixed panel: each says HOW MUCH of each marker it expresses.
    """
    types = {}
    for i, on in enumerate(struct["support"]):
        tname = name_fmt.format(i)
        expr = {}
        for n in on:
            # DAPI is a DNA stain: every nucleus takes it, and at a much more consistent level
            # than an antibody marker, so its expression range is tight around 1.
            elo, ehi = (0.85, 1.15) if n == DAPI_NAME else (lo, hi)
            expr[n] = P(0.0, 2.0, .05, float(rng.uniform(elo, ehi)), f"expr {n}",
                        comment=f"expression level of {n} in this cell type; "
                                f"0 = negative, 1 = the panel's nominal brightness")
        types[tname] = CellType(name=tname, Color="black",
                                Geometry=copy.deepcopy(geom) if geom is not None
                                else CellGeometry(RADIUS=8.0, ROUGH=0.15, ELONG=1.4,
                                                  NUC_FRAC=0.40, RIM=0.0, NUC_OFFSET=0.8),
                                Expression=expr)
    return types
def generate_world(struct, rng, n_markers=3, n_types=2, expr_frac=0.6, struct_seed=0):
    """The objects whose parameters are fitted.
    `struct` fixes the theta layout across a dataset; 
    `rng` draws this sample's values. 
    """
    cell = CellGeometry(RADIUS=8.0, ROUGH=0.15, ELONG=1.4, NUC_FRAC=0.40, RIM=0.0,
                        NUC_OFFSET=0.8)
    panel = build_panel(struct, rng)
    types = build_cell_types(struct, rng, geom=cell)
    # Use dirichlet to sample simplex with equal likelihood of ever combination
    fractions = rng.dirichlet(np.ones(len(types))) if len(types) > 1 else np.array([1.0])
    return {"PANEL": panel, "TYPES": types, "TG": TissueGeometry(),
            # The Detector is predefined and not trainable 
            # (We know it from the real device)
            "DET": Detector(**cfg.DETECTOR),
            "CELL": cell, "FRACTIONS": list(fractions)}
    
    
def render_frame(world, seed, pool_bank=POOL_BANK):
    """theta (already written into `world`) + a tape seed -> one uint16 detector frame.

    The image has one channel per PANEL marker, whatever the cell types do -- a marker that no
    type expresses is a dark channel, which is exactly what a real panel gives you.

    `pool_bank` caps the tape's frozen fields. One field per (marker, component) is exact but
    costs ~1 GB at 40 markers on a tissue volume; beyond the bank, fields are reused at a
    deterministic shift (see render._pool_field).
    """
    panel, types, tg, det = world["PANEL"], world["TYPES"], world["TG"], world["DET"]
    opt = Optics(um_per_px=UM_PER_VOX, um_per_pz=UM_PER_VOX * Z_RATIO)
    need = panel.n_pools()
    n_pools = need if not pool_bank else min(int(pool_bank), need)

    tape = Tape(seed=int(seed), size=SIZE, K=K)
    tape.draw(tile=TILE, n_cand=N_CAND, Pool=n_pools)

    shape = TISSUE_VOL
    tape.draw3d(vol=CELL_VOL, n_cand=N_CAND, Pool=n_pools, l_min=L_MIN, L=L)
    tape.drawTissue(shape=shape, n_cand=N_CAND, Pool=n_pools)
    vols, *_ = build_tissue(tape=tape, TG=tg, shape=shape, base_geom=world["CELL"],
                            spacing=SPACING, Panel=panel, CellTypes=types,
                            Fractions=world["FRACTIONS"], um_per_vox=UM_PER_VOX,
                            L=L, l_min=L_MIN, pool_bank=pool_bank)
    
    subs = {k: kryostat(vols[k], opt) for k in panel.names}
    psf = np.stack([psf_project(v, z, opt, panel.Markers[k].fluorophore)
                    for k, (v, z) in subs.items()], -1)
    tape.drawSensor(shape=(shape[1], shape[2], len(panel.names)))
    adu = detector(psf, panel, det, opt, tape, panel.names, quantise=True)
    return np.clip(adu, 0, 65535).astype(np.uint16)


def draw_one(i):
    """Sample index -> (image, theta).
    The PANEL AND THE CELL TYPES ARE REDRAWN PER SAMPLE.
    """
    cfg = _W["cfg"]
    # same STRUCTURE, new RNG every sample
    world = generate_world(struct=cfg["struct"], rng = np.random.default_rng(_W["seed"] + 500_000 + i))
    paths = cfg["paths"]
    rng = np.random.default_rng(_W["seed"] + i)
    u = sample_prior(rng, paths)
    set_vector(world, paths, u)
    img = render_frame(world, seed=_W["seed"] + 10_000_000 + i,
                       pool_bank=cfg["pool_bank"])
    return i, img, u.astype(np.float32)

def worker_samples(n, seed, keep, run_cfg, workers):
    """Yield (index, image, theta) for every sample, from one process or from a pool."""
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(seed, keep, run_cfg)) as ex:
            yield from ex.map(draw_one, range(n), chunksize=1)
    else:
        _init_worker(seed, keep, run_cfg)
        yield from map(draw_one, range(n))
        
def make_dataset(a, seed, pool_bank=POOL_BANK):
    struct = panel_structure(a.n_markers, a.n_types, a.expr_frac, struct_seed=seed)
    ref = generate_world(struct, rng = np.random.default_rng(0))
    paths = fitted_paths(ref, keep=a.keep)
    lo, hi = bounds(ref, paths)
    run_cfg = dict(struct=struct, pool_bank=pool_bank, paths=paths)
    
    frame_shape = (TISSUE_VOL[1], TISSUE_VOL[2], len(ref["PANEL"].names))
    stem = re.sub(r"\.(meta\.npz|npz|npy)$", "", a.out)
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)

    print(f"generating {a.n} samples | {a.n_markers} markers x {a.n_types} cell types "
          f"| {len(paths)} fitted parameters | {a.workers} worker(s)")
    need = ref["PANEL"].n_pools()
    
    # Write frames directly into the npy to reduce RAM requirements    
    images = np.lib.format.open_memmap(stem + ".npy", mode="w+", dtype=np.uint16,
                                       shape=(a.n,) + frame_shape)
    theta = np.zeros((a.n, len(paths)), np.float32)
    sat = np.zeros(a.n)
    t0 = time.time()

    for k, (i, img, u) in enumerate(worker_samples(a.n, seed, a.keep, run_cfg, a.workers), 1):
        images[i] = img
        theta[i] = u
        sat[i] = float((img >= cfg.ADU_MAX).mean())
        if k % max(1, a.n // 20) == 0 or k == a.n:
            el = time.time() - t0
            print(f"  {k}/{a.n}  {el:.0f}s elapsed, {el / k * (a.n - k):.0f}s left", flush=True)

    images.flush()
    del images
    np.savez(stem + ".meta.npz", theta=theta, paths=np.array(paths), lo=lo, hi=hi,
             seed=seed, names=np.array(ref["PANEL"].names),
             dyes=np.array([ref["PANEL"].fluorophores[k] for k in ref["PANEL"].names]))
    
def make_argparse():
    
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="../data/example.npz")
    ap.add_argument("--n-markers", type=int, default=30, dest="n_markers",
                   help="markers in the panel = channels in the image")
    ap.add_argument("--n-celltypes", type=int, default=2, dest="n_types")
    ap.add_argument("--expr-frac", type=float, default=0.6, dest="expr_frac",
                   help="expected fraction of the panel a cell type is positive for")
    ap.add_argument("--keep", nargs="*", default=KEEP_DEFAULT)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    return a

if __name__ == "__main__":
    seed = 42
    a = make_argparse()
    make_dataset(a, seed)