"""Exact 2D label export (Phase 2).

The generator already knows everything about the scene it rendered -- which cell owns each voxel,
which structure each voxel sits in, every cell's type and geometry. This module collapses that
3D truth to the 2D frame the model sees, through the SAME optics that made the image, so the labels
line up with the pixels. Nothing here is segmented out of the image: every array is a deterministic
readout of the generator's own state.

Outputs per frame:
  inst         uint32 [H, W]  cell instance ids (same ids as the GeoJSON polygons)
  nuc          uint8  [H, W]  1 = nucleus pixel of a VISIBLE cell
  type_map     uint8  [H, W]  0 = background, 1..K_true = type (sample order), 255 = ignore
  context_map  uint8  [H, W]  0 = background tissue, 1.. = structure kind (cfg.CONTEXT_KINDS)
  cells        structured array, one row per present instance (see CELL_DTYPE)

A cell is VISIBLE if its 2D area >= cfg.VIS["min_area_px"]; a type is PRESENT if it has at least
cfg.VIS["min_visible_cells"] visible cells. Pixels of invisible cells and of non-present types get
the ignore label, so the loss is never charged where there is too little evidence.
"""
import numpy as np

from optics import kryostat, mask_collapse
from artifacts import lift_slab, extend_z
import config as cfg

CELL_DTYPE = np.dtype([
    ("id",         np.int32),
    ("type_idx",   np.int32),   # 1..K_true, index into the sample's type order (0 = n/a)
    ("context",    np.int32),   # 0 = background tissue, 1.. = cfg.CONTEXT_KINDS
    ("visible",    np.bool_),
    ("area_px",    np.int32),
    ("cy",         np.float32),
    ("cx",         np.float32),
    ("radius_vox", np.float32),
    ("elong",      np.float32),
    ("polar_deg",  np.float32),
    ("azim_deg",   np.float32),
])


def _collapse(vol3d, opt, shift, n_ext, mask_pct):
    """Section a 3D label volume and collapse it to 2D through the optics, matching exactly how the
    cell mask is produced in render_frame (same kryostat + detachment lift)."""
    sub, sub_z = kryostat(vol3d, opt)
    if shift is not None:
        sub, sub_z = lift_slab(sub, shift, n_ext), extend_z(sub_z, n_ext, opt.um_per_pz)
    out, _ = mask_collapse(sub, sub_z, opt, mask_pct=mask_pct)
    return out


def _context_of_index(struct_idx, arch, context_kinds):
    """structure instance index (into arch.Structures) -> context id (0 bg, 1.. kind)."""
    if struct_idx < 0 or arch is None or struct_idx >= len(arch.Structures):
        return 0
    kind = getattr(arch.Structures[struct_idx], "kind", None)
    return (context_kinds.index(kind) + 1) if kind in context_kinds else 0


def build_frame_labels(cell2d, nuc_labels3d, field, types, info, type_names, opt,
                       shift=None, n_ext=0, context_kinds=None,
                       min_area_px=None, min_visible_cells=None):
    """Collapse the generator's 3D truth to the 2D label set. `cell2d` is the already-collapsed
    instance map from render_frame (reused so `inst` matches the GeoJSON exactly). `type_names` is
    the sample's canonical type order (list(CellTypes)); `types` maps instance id -> type name."""
    context_kinds = cfg.CONTEXT_KINDS if context_kinds is None else context_kinds
    min_area_px = cfg.VIS["min_area_px"] if min_area_px is None else min_area_px
    min_visible_cells = cfg.VIS["min_visible_cells"] if min_visible_cells is None else min_visible_cells
    arch = info.get("arch")

    inst = np.ascontiguousarray(cell2d).astype(np.uint32)
    H, W = inst.shape
    type_index = {name: i + 1 for i, name in enumerate(type_names)}   # 1..K_true

    # --- per-cell geometry / visibility, from the exact scene state ---
    ids = [int(n) for n in np.unique(inst) if n != 0]
    geoms, centres_vox = info["geoms"], info["centres_vox"]
    labels3d = field["labels"] if (field is not None) else None

    rows, visible_of, typeidx_of = [], {}, {}
    visible_count = {}
    for n in ids:
        m = inst == n
        area = int(m.sum())
        vis = area >= min_area_px
        tname = types.get(n)
        tidx = type_index.get(tname, 0)
        g = geoms.get(n)
        # context from the cell's own centre voxel in the structure-label volume
        ctx = 0
        if labels3d is not None and n in centres_vox:
            cx, cy, cz = centres_vox[n]
            nz, ny, nx = labels3d.shape
            si = int(labels3d[int(np.clip(cz, 0, nz - 1)),
                              int(np.clip(cy, 0, ny - 1)),
                              int(np.clip(cx, 0, nx - 1))])
            ctx = _context_of_index(si, arch, context_kinds)
        ys, xs = np.nonzero(m)
        rows.append((n, tidx, ctx, vis, area,
                     float(ys.mean()) if area else 0.0, float(xs.mean()) if area else 0.0,
                     float(g.RADIUS.v) if g is not None else 0.0,
                     float(g.ELONG.v) if g is not None else 0.0,
                     float(g.POLAR_DEG.v) if g is not None else 0.0,
                     float(g.AZIM_DEG.v) if g is not None else 0.0))
        visible_of[n] = vis
        typeidx_of[n] = tidx
        if vis and tname is not None:
            visible_count[tname] = visible_count.get(tname, 0) + 1
    cells = np.array(rows, dtype=CELL_DTYPE) if rows else np.empty(0, dtype=CELL_DTYPE)

    # a type is present if it has enough visible cells; non-present -> ignore in type_map
    present = np.array([visible_count.get(name, 0) >= min_visible_cells for name in type_names],
                       dtype=bool)
    present_type = {name for name, p in zip(type_names, present) if p}

    # --- type_map: bg 0, type 1..K_true for visible cells of present types, else ignore ---
    type_map = np.zeros((H, W), np.uint8)
    for n in ids:
        tname = types.get(n)
        if visible_of[n] and tname in present_type:
            type_map[inst == n] = typeidx_of[n]
        else:
            type_map[inst == n] = cfg.IGNORE_LABEL

    # --- nucleus mask (visible cells only) ---
    # cell and nucleus footprints are collapsed independently through the PSF, so at a boundary
    # they can disagree by a pixel. A nucleus pixel is only honest where the collapsed nucleus id
    # equals the collapsed CELL id at that pixel (the nucleus sits inside its own cell's body), so
    # require nuc2d == inst; that also guarantees nuc is a subset of the visible-cell footprint.
    nuc = np.zeros((H, W), np.uint8)
    if nuc_labels3d is not None and nuc_labels3d.any():
        nuc2d = _collapse(nuc_labels3d, opt, shift, n_ext, mask_pct=0.95)
        vis_ids = np.array([n for n in ids if visible_of[n]], dtype=nuc2d.dtype)
        if vis_ids.size:
            nuc[(nuc2d > 0) & (nuc2d == inst.astype(nuc2d.dtype)) & np.isin(nuc2d, vis_ids)] = 1

    # --- context_map: structure kind per pixel ---
    context_map = np.zeros((H, W), np.uint8)
    if labels3d is not None and (labels3d >= 0).any():
        # shift so structure index 0 is not swallowed as background by mask_collapse (bg -1 -> 0)
        shifted = np.where(labels3d >= 0, labels3d.astype(np.int32) + 1, 0)
        struct2d = _collapse(shifted, opt, shift, n_ext, mask_pct=0.90)
        for v in np.unique(struct2d):
            if v > 0:
                context_map[struct2d == v] = _context_of_index(int(v) - 1, arch, context_kinds)

    return dict(inst=inst, nuc=nuc, type_map=type_map, context_map=context_map,
                cells=cells, type_names=list(type_names), present=present,
                context_kinds=list(context_kinds))
