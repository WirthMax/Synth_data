import argparse, copy, json, time, os, re
import numpy as np

from scene import build_tissue_V2, artifact_priority, build_training_mask
from render import render_image, to_rgb
from tape import Tape
from plot import plot_surface_xyz_inline, plot_surface_xyz_html, plot_surface_rgb_html, ortho, ortho_rgb, tau_cmap, NormalizeData
from optics import kryostat, psf_project, mask_collapse, detector
from artifacts import build_artifacts, detachment_map, lift_slab, extend_z, lift_project
from geojson_export import build_frame_geojson
from parameter import (P, CellGeometry, MarkerPanel, PanelMarker, CellType, Detector,
                       BlobNoise, ClusterNoise, NetworkNoise, FibreNoise, SheetNoise,
                       Optics, TissueGeometry, Vasculature, Ordered, TissueArchitecture,
                       Artifacts, dapi_marker, DEVICE_DETECTOR,
                       NOISE_KINDS, narrow, DEFAULT_PIN,
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
    arch = architecture_structure(n_types, struct_seed=struct_seed)
    return dict(markers=markers, support=support, arch=arch)


def architecture_structure(n_types, struct_seed=0, max_structures=3, name_fmt="T{:03d}"):
    """Which tissue structures exist and which cell types each is affiliated with -- drawn once
    per dataset, analogous to the marker/support structure above. Individual PARAMETER VALUES
    (size, direction, aspect, per-type affinity strength...) are redrawn per sample by the normal
    fitted_paths/set_vector theta machinery (see cfg.KEEP's `^ARCH\\.` pattern); only this SHAPE
    (structure count, kind, which types each Profile references) has to stay fixed, since every
    frame in a dataset shares one theta layout. build_tissue_V2 requires at least one structure.
    """
    r = np.random.default_rng(struct_seed + 7_919)  # offset from panel_structure's own draws
    type_names = [name_fmt.format(i) for i in range(n_types)]
    n = int(r.integers(1, max_structures + 1))
    structures = []
    for i in range(n):
        kind = "vessel" if r.random() < 0.5 else "ordered"
        chosen = [t for t in type_names if r.random() < 0.6] or [str(r.choice(type_names))]
        structures.append(dict(kind=kind, name=f"s{i}", profile_types=chosen))
    return structures


def build_architecture_instance(arch_struct):
    """Fresh TissueArchitecture matching `arch_struct`'s shape. Field VALUES are whatever the
    class declares as default -- set_vector overwrites every path that made it into theta right
    after this is called, same as CellGeometry/PanelMarker fields built elsewhere in this file.

    FRAC is the one field that can't be left at its class-declared [0, 1] range once it's part of
    theta: tissue_structures.build_architecture requires every structure's FRAC to sum to <= 1
    (they're shares of the tissue, not independent), and structures are sampled independently.
    Narrowing each one's ceiling to a share of 0.9 keeps every possible per-frame draw valid while
    still giving each structure real value variety.

    A vessel additionally carves a lumen (void, non-tissue) out of its own territory, sized by
    WALL_IN -- at WALL_IN's default upper bound (0.9) combined with a large FRAC, the lumen alone
    can exceed TG.COVER's tissue-voxel target for the WHOLE frame (observed: a single vessel's
    void ate ~26% of the frame against a 75% COVER target, an infeasible carve). Vessels get half
    the FRAC ceiling ordered bundles do, and WALL_IN is capped at 0.5 (lumen <= 25% of the
    vessel's own territory), so even every structure in a dataset being a maxed-out vessel leaves
    COVER comfortable headroom.
    """
    structures = []
    frac_hi = 0.9 / max(len(arch_struct), 1)
    for s in arch_struct:
        cls = Vasculature if s["kind"] == "vessel" else Ordered
        st = cls(name=s["name"], Profile={t: 0.0 for t in s["profile_types"]})
        st.FRAC = narrow(st.FRAC, st.FRAC.lo, frac_hi * (0.5 if s["kind"] == "vessel" else 1.0))
        if s["kind"] == "vessel":
            st.WALL_IN = narrow(st.WALL_IN, st.WALL_IN.lo, min(st.WALL_IN.hi, 0.5))
        structures.append(st)
    arch = TissueArchitecture(Structures=structures)
    # Ordered's class default ANISO is 3.0 (vs. BaseStructure/Vasculature's 1.0), so mixing
    # structure kinds gives non-uniform compliance regardless of theta. CONTACT>0 is what turns
    # that mismatch into tissue_structures' compliance-based pressure solve, which doesn't
    # reliably converge -- pin CONTACT at 0 (the additively-weighted rule) so that solve path
    # never engages. ARCH_PIN (gen_dataset.make_dataset) keeps theta from re-enabling it.
    arch.CONTACT = narrow(arch.CONTACT, 0.0, 0.0)
    return arch

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
def generate_world(struct, rng, n_markers=3, n_types=2, expr_frac=0.6, struct_seed=0,
                   with_artifacts=False, max_aggregates=None, max_fussels=None,
                   max_detach_cover=None):
    """The objects whose parameters are fitted.
    `struct` fixes the theta layout across a dataset;
    `rng` draws this sample's values.

    `max_aggregates`/`max_fussels` cap AR.Aggregate.N/AR.Fussel.N's upper bound (their class
    declarations allow up to 32/4) -- narrowing rather than clamping post-hoc, so the capped
    range is what theta's `lo`/`hi` actually record and what sample_prior/set_vector draw from.

    `max_detach_cover` caps AR.Detachment.COVER the same way (class declaration allows up to
    1.0). build_training_mask (scene.py) drops a cell -- as a WHOLE object -- once more than
    half its footprint sits in a detached/lifted region, and that region's own area is
    calibrated to be exactly `COVER`-fraction of the frame's tissue (detachment_map,
    artifacts.py). Because cells here sit in tight tissue nests rather than spread evenly
    across the frame, whether a nest lands inside or outside that region is close to binary
    per frame once COVER is past a small value -- the notebook's own worked example
    (3d_cell_generation.ipynb, `Detachment(COVER=0.10, ...)`) is the only value anyone has
    actually looked at, and the real dataset's fraction of fully-empty frames jumps from 0%
    to 31% just crossing COVER=0.10. Leave this uncapped only if frames with heavily/fully
    detached tissue (no usable ground truth at all) are an intentional part of the dataset.
    """
    cell = CellGeometry(RADIUS=8.0, ROUGH=0.15, ELONG=1.4, NUC_FRAC=0.40, RIM=0.0,
                        NUC_OFFSET=0.8)
    panel = build_panel(struct, rng)
    types = build_cell_types(struct, rng, geom=cell)
    # Use dirichlet to sample simplex with equal likelihood of ever combination
    fractions = rng.dirichlet(np.ones(len(types))) if len(types) > 1 else np.array([1.0])
    world = {"PANEL": panel, "TYPES": types, "TG": TissueGeometry(),
            "ARCH": build_architecture_instance(struct["arch"]),
            # The Detector is predefined and not trainable
            # (We know it from the real device)
            "DET": Detector(**cfg.DETECTOR),
            "CELL": cell, "FRACTIONS": list(fractions)}
    if with_artifacts:
        ar = Artifacts()
        if max_aggregates is not None:
            ar.Aggregate.N = narrow(ar.Aggregate.N, ar.Aggregate.N.lo, float(max_aggregates))
        if max_fussels is not None:
            ar.Fussel.N = narrow(ar.Fussel.N, ar.Fussel.N.lo, float(max_fussels))
        if max_detach_cover is not None:
            ar.Detachment.COVER = narrow(ar.Detachment.COVER, ar.Detachment.COVER.lo,
                                         float(max_detach_cover))
        world["AR"] = ar
    return world
    
    
def render_frame(world, seed, pool_bank=POOL_BANK, with_artifacts=False):
    """theta (already written into `world`) + a tape seed -> (image, training mask, geojson).

    The image has one channel per PANEL marker, whatever the cell types do -- a marker that no
    type expresses is a dark channel, which is exactly what a real panel gives you.

    `pool_bank` caps the tape's frozen fields. One field per (marker, component) is exact but
    costs ~1 GB at 40 markers on a tissue volume; beyond the bank, fields are reused at a
    deterministic shift (see render._pool_field).

    The mask/geojson side always runs (that's the point of generating this dataset for Cellpose
    training), but with_artifacts=False leaves art2d/detach_gate at zero, which is a no-op for
    artifact_priority/build_training_mask -- one code path either way.
    """
    panel, types, tg, det, arch = (world["PANEL"], world["TYPES"], world["TG"], world["DET"],
                                   world["ARCH"])
    opt = Optics(um_per_px=UM_PER_VOX, um_per_pz=UM_PER_VOX * Z_RATIO)
    need = panel.n_pools()
    n_pools = need if not pool_bank else min(int(pool_bank), need)

    tape = Tape(seed=int(seed), size=SIZE, K=K)
    tape.draw(tile=TILE, n_cand=N_CAND, Pool=n_pools)

    shape = TISSUE_VOL
    tape.draw3d(vol=CELL_VOL, n_cand=N_CAND, Pool=n_pools, l_min=L_MIN, L=L)
    n_struct = len(arch.Structures)
    tape.drawTissueDir(shape=shape, n_cand=N_CAND, Pool=n_pools, n_struct=n_struct,
                       n_inst=32 * n_struct)
    if with_artifacts:
        tape.drawArtifacts(shape=shape, **cfg.ARTIFACTS)

    vols, labels, _nuc_labels, _tau_img, _types_by_id, info = build_tissue_V2(
        tape=tape, ARCH=arch, TG=tg, shape=shape, base_geom=world["CELL"],
        spacing=SPACING, Panel=panel, CellTypes=types, Fractions=world["FRACTIONS"],
        um_per_vox=UM_PER_VOX, L=L, l_min=L_MIN, pool_bank=pool_bank)

    sub, sub_z = kryostat(labels, opt)

    shift, n_ext = None, 0
    art_labels, detach_gate = None, None
    if with_artifacts:
        vols_art = {k: v.copy() for k, v in vols.items()}
        art = build_artifacts(vols=vols_art, tape=tape, AR=world["AR"], opt=opt, shape=shape,
                              panel=panel, um_per_vox=UM_PER_VOX, spacing=SPACING,
                              geom=world["CELL"], tissue_support=info["support"], L=L, l_min=L_MIN,
                              pool_bank=pool_bank)
        detach = detachment_map(tape, world["AR"].Detachment, info["support"], SPACING,
                                UM_PER_VOX, opt.um_per_pz)
        shift = detach["shift"]
        n_ext = int(shift.max())
        vols = vols_art
        art_labels = art["labels"]
        detach_gate = detach["gate"]
        sub, sub_z = lift_slab(sub, shift, n_ext), extend_z(sub_z, n_ext, opt.um_per_pz)

    subs = {k: kryostat(vols[k], opt) for k in panel.names}
    if with_artifacts:
        psf = np.stack([lift_project(v, z, shift, n_ext, opt, panel.Markers[k].fluorophore)
                        for k, (v, z) in subs.items()], -1)
    else:
        psf = np.stack([psf_project(v, z, opt, panel.Markers[k].fluorophore)
                        for k, (v, z) in subs.items()], -1)
    tape.drawSensor(shape=(shape[1], shape[2], len(panel.names)))
    adu = detector(psf, panel, det, opt, tape, panel.names, quantise=True)
    img = np.clip(adu, 0, 65535).astype(np.uint16)

    cell2d, _ = mask_collapse(sub, sub_z, opt, mask_pct=0.95)
    if with_artifacts:
        sub_art, z_art = kryostat(art_labels, opt)
        sub_art, z_art = lift_slab(sub_art, shift, n_ext), extend_z(z_art, n_ext, opt.um_per_pz)
        art2d, _ = mask_collapse(sub_art, z_art, opt, mask_pct=0.90)
    else:
        art2d = np.zeros_like(cell2d)
        detach_gate = np.zeros(cell2d.shape, np.float32)

    combined2d = artifact_priority(cell2d, art2d)
    mask = build_training_mask(combined2d, detach_gate)
    geojson = json.dumps(build_frame_geojson(cell2d, art2d))
    return img, mask.astype(np.int32), geojson


def draw_one(i, max_retries=5):
    """Sample index -> (image, theta, training mask, geojson).
    The PANEL AND THE CELL TYPES ARE REDRAWN PER SAMPLE.

    A randomly-sampled architecture is occasionally geometrically infeasible (e.g. a vessel's
    lumen too big for TG.COVER's tissue budget, or the pressure solve landing on a degenerate
    seed layout) -- tissue_structures/pressure_pack raise ValueError for exactly this class of
    "this particular draw doesn't work" problem, distinct from an actual bug. At dataset scale,
    aborting the whole batch on one unlucky draw is worse than redrawing that one sample with a
    perturbed seed, so retry a bounded number of times before giving up on index i.
    """
    cfg = _W["cfg"]
    paths = cfg["paths"]
    last_err = None
    for attempt in range(max_retries):
        offset = attempt * 1_000_000_000
        try:
            # same STRUCTURE, new RNG every sample (and every retry)
            world = generate_world(struct=cfg["struct"],
                                   rng=np.random.default_rng(_W["seed"] + 500_000 + i + offset),
                                   with_artifacts=cfg["with_artifacts"],
                                   max_aggregates=cfg["max_aggregates"],
                                   max_fussels=cfg["max_fussels"],
                                   max_detach_cover=cfg["max_detach_cover"])
            rng = np.random.default_rng(_W["seed"] + i + offset)
            u = sample_prior(rng, paths)
            set_vector(world, paths, u)
            img, mask, geojson = render_frame(world, seed=_W["seed"] + 10_000_000 + i + offset,
                                              pool_bank=cfg["pool_bank"],
                                              with_artifacts=cfg["with_artifacts"])
            return i, img, u.astype(np.float32), mask, geojson
        except ValueError as e:
            last_err = e
            print(f"  [sample {i}] attempt {attempt + 1}/{max_retries} failed ({e}); retrying "
                 f"with a perturbed seed", flush=True)
    raise RuntimeError(f"sample {i} failed after {max_retries} attempts") from last_err

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
    ref = generate_world(struct, rng=np.random.default_rng(0), with_artifacts=a.with_artifacts,
                         max_aggregates=a.max_aggregates, max_fussels=a.max_fussels,
                         max_detach_cover=a.max_detach_cover)
    # STIFF/ANISO/CONTACT govern tissue_structures' pressure-pack solve, which can fail to
    # converge or blow up under independently-sampled structures (divergent compliance ratios) --
    # pin them out of theta so every structure stays uniformly stiff/isotropic and CONTACT's
    # compliance-blend path (which only engages when compliance actually varies) never activates.
    ARCH_PIN = (r"\.STIFF$", r"\.ANISO$", r"^ARCH\.CONTACT$")
    paths = fitted_paths(ref, pin=DEFAULT_PIN + ARCH_PIN, keep=a.keep)
    lo, hi = bounds(ref, paths)
    run_cfg = dict(struct=struct, pool_bank=pool_bank, paths=paths, with_artifacts=a.with_artifacts,
                   max_aggregates=a.max_aggregates, max_fussels=a.max_fussels,
                   max_detach_cover=a.max_detach_cover)

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
    # masks/geojson are single-channel and comparatively tiny, so unlike images they're just
    # collected in RAM and written once at the end -- one key pair PER FRAME (not one big stacked
    # array), so a later reader can pull exactly the frames its current batch needs:
    # NpzFile.__getitem__ decompresses only the member asked for.
    mask_dict = {}
    t0 = time.time()

    for k, (i, img, u, mask, geojson) in enumerate(
            worker_samples(a.n, seed, a.keep, run_cfg, a.workers), 1):
        images[i] = img
        theta[i] = u
        mask_dict[f"mask_{i:06d}"] = mask
        mask_dict[f"geojson_{i:06d}"] = np.array(geojson)
        sat[i] = float((img >= cfg.ADU_MAX).mean())
        if k % max(1, a.n // 20) == 0 or k == a.n:
            el = time.time() - t0
            print(f"  {k}/{a.n}  {el:.0f}s elapsed, {el / k * (a.n - k):.0f}s left", flush=True)

    images.flush()
    del images
    np.savez_compressed(stem + ".masks.npz", n_frames=a.n, with_artifacts=a.with_artifacts,
                        **mask_dict)
    np.savez(stem + ".meta.npz", theta=theta, paths=np.array(paths), lo=lo, hi=hi,
             seed=seed, names=np.array(ref["PANEL"].names),
             dyes=np.array([ref["PANEL"].fluorophores[k] for k in ref["PANEL"].names]),
             with_artifacts=a.with_artifacts,
             max_aggregates=-1 if a.max_aggregates is None else a.max_aggregates,
             max_fussels=-1 if a.max_fussels is None else a.max_fussels,
             max_detach_cover=-1 if a.max_detach_cover is None else a.max_detach_cover)


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
    ap.add_argument("--with-artifacts", action="store_true", dest="with_artifacts",
                   help="generate debris/aggregate/detachment artifacts, bake them into the "
                        "image, and include artifact-rate params in theta")
    ap.add_argument("--max-aggregates", type=float, default=None, dest="max_aggregates",
                   help="cap AR.Aggregate.N (random aggregate count per frame stays in "
                        "[0, this]); only meaningful with --with-artifacts. Default: the "
                        "class's own bound (32)")
    ap.add_argument("--max-fussels", type=float, default=None, dest="max_fussels",
                   help="cap AR.Fussel.N (random fussel count per frame stays in [0, this]); "
                        "only meaningful with --with-artifacts. Default: the class's own "
                        "bound (4)")
    ap.add_argument("--max-detach-cover", type=float, default=None, dest="max_detach_cover",
                   help="cap AR.Detachment.COVER (random per-frame tissue-detachment fraction "
                        "stays in [0.05, this]); only meaningful with --with-artifacts. "
                        "build_training_mask drops a cell's whole footprint once it's majority "
                        "inside a detached region, and because cells sit in tight tissue nests "
                        "rather than spread evenly, whether a frame's cells fall inside that "
                        "region is close to a coin flip once COVER is past a small value -- "
                        "the only value ever visually checked in the notebook is 0.10. Default: "
                        "the class's own bound (1.0), i.e. frames can go fully detached/empty")
    a = ap.parse_args()
    return a

if __name__ == "__main__":
    seed = 42
    a = make_argparse()
    make_dataset(a, seed)