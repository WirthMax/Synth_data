"""Interactive 3D viewing in the browser -- the counterpart to `plot3d.py`.

`plot3d` renders static matplotlib figures (ortho slices, montage, MIP, shaded surface): good for
a report, but you cannot rotate them and they cannot show what is INSIDE a cell. This module
builds plotly scenes instead, which render inline in Jupyter and export to a single standalone
HTML file.

Two modes, one entry point (`scene3d`):

  mode="surface"   marching-cubes meshes of the cell and nuclear surfaces, optionally coloured
                   per triangle by the marker RGB (or per instance by phenotype). Cheap, and it
                   scales straight to a tissue tile with hundreds of cells.
  mode="volume"    raymarched `go.Volume` per marker channel, so the INTERIOR 3D structure --
                   filaments, vesicles, the nucleoplasmic mesh -- is directly visible. This is
                   the mode that shows what the voxel fill in `render3d.py` actually bought.

Design notes worth keeping in mind:

* We march the ANALYTIC implicit fields, not the boolean masks: `d` at level 1.0 for the plasma
  membrane and `tau` at level 0.0 for the nuclear envelope. `cell_fields_3d` guarantees the
  labelled nuclear surface and the tau = 0 level set are the same surface, so the mesh cannot
  disagree with the mask -- and the result is sub-voxel accurate instead of stair-stepped.
* Volumes are indexed (z, y, x); plotly gets x <- col, y <- row, z <- depth. That permutation is
  ODD, so it reverses handedness -- get the triangle winding wrong and flat shading lights every
  surface from the inside. `_orient` decides it by measuring, not by assuming (see there).
* Physical units: pass an `Optics` and everything is in microns, with the volume centred on z = 0
  (the `plane_heights` convention), so `focal_um` and `section_slab(centre_um=...)` are directly
  plottable. Without one, everything is in voxels.

plotly and scikit-image are imported LAZILY, so `import viewer3d` (and therefore
`from viewer3d import *` in `__init__.py`) keeps working in an environment that has neither.
"""

import warnings

import numpy as np
from scipy import ndimage as ndi


# --------------------------------------------------------------------- lazy dependencies

def _go():
    try:
        import plotly.graph_objects as go
    except ImportError as e:                                     # pragma: no cover
        raise ImportError("viewer3d needs plotly >= 6:  pip install 'plotly>=6'") from e
    return go


def _mc():
    try:
        from skimage.measure import marching_cubes
    except ImportError as e:                                     # pragma: no cover
        raise ImportError("viewer3d needs scikit-image:  pip install scikit-image") from e
    return marching_cubes


# --------------------------------------------------------------------- constants

#: plotly's default qualitative colorway -- the previous project's viewer used exactly this for
#: per-cell phenotype colour, so instance colouring here reproduces it.
PLOTLY_COLORWAY = ("#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
                   "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52")

DEFAULT_CAMERA = (1.6, 1.6, 0.9)

#: Single-hue scales, transparent at the low end so overlaid channels composite instead of
#: occluding each other. `go.Volume` has no RGB mode, so three markers = three traces.
CHANNEL_SCALES = {
    "red":     [[0.0, "rgba(0,0,0,0)"], [1.0, "rgb(255,60,60)"]],
    "green":   [[0.0, "rgba(0,0,0,0)"], [1.0, "rgb(60,255,90)"]],
    "blue":    [[0.0, "rgba(0,0,0,0)"], [1.0, "rgb(80,140,255)"]],
    "magenta": [[0.0, "rgba(0,0,0,0)"], [1.0, "rgb(255,80,220)"]],
    "cyan":    [[0.0, "rgba(0,0,0,0)"], [1.0, "rgb(60,230,230)"]],
    "yellow":  [[0.0, "rgba(0,0,0,0)"], [1.0, "rgb(255,220,60)"]],
}

_NUC_COLOUR = "#111820"
_FOCAL_COLOUR = "#FFD700"
_BOX_COLOUR = "#444"


# --------------------------------------------------------------------- meshing

def mesh_from_field(field, level=0.5, inside="high", step_size=1, spacing=(1.0, 1.0, 1.0),
                    origin=(0.0, 0.0, 0.0), pad=True, clip_z=None, smooth=0.0):
    """Marching cubes on one (z, y, x) field.

    Returns ``(verts_plot, faces, normals, verts_idx)``, or ``(None, None, None, None)`` if the
    level set is empty (an all-background volume makes skimage raise, which is not useful here).

    `verts_plot` is in plot units (`verts_idx * spacing + origin`); `verts_idx` is kept in voxel
    index units because that is the frame the marker volumes are sampled in. Marching is always
    done at unit spacing and scaled afterwards, which sidesteps any ambiguity about the axis
    order skimage's own `spacing=` argument expects.

    `inside`
        "high" for masks (inside = 1 > outside = 0) -> gradient_direction "descent".
        "low"  for `d` at level 1.0 or `tau` at level 0.0, where inside is the SMALLER value
               -> gradient_direction "ascent". Getting this wrong inverts the normals.
    `pad`
        Zero-pad by one voxel first. NOT optional in practice: an object touching the array
        border produces an OPEN surface, which at opacity 0.55 reads as a hollow shell.
    `clip_z`
        (lo, hi) in plot units; clamps the depth coordinate after offsetting, so a cell cut by a
        section ends flush with the section box instead of overshooting by the half-voxel the
        marching cubes lattice adds.
    """
    marching_cubes = _mc()
    f = np.asarray(field, np.float32)
    if smooth > 0:
        f = ndi.gaussian_filter(f, smooth)
    # the pad value has to sit on the OUTSIDE of the level set, whichever side that is
    if pad:
        outer = float(f.min() - 1.0) if inside == "high" else float(f.max() + 1.0)
        f = np.pad(f, 1, constant_values=outer)
    if not (f.min() <= level <= f.max()):
        return None, None, None, None
    try:
        verts, faces, normals, _ = marching_cubes(
            f, level, step_size=int(step_size), allow_degenerate=False,
            gradient_direction="descent" if inside == "high" else "ascent")
    except (ValueError, RuntimeError):
        return None, None, None, None
    if len(faces) == 0:
        return None, None, None, None
    if pad:
        verts = verts - 1.0
    verts_plot = verts * np.asarray(spacing, np.float32) + np.asarray(origin, np.float32)
    if clip_z is not None:
        verts_plot[:, 0] = np.clip(verts_plot[:, 0], clip_z[0], clip_z[1])
    return (verts_plot.astype(np.float32), faces.astype(np.int32),
            normals.astype(np.float32), verts.astype(np.float32))


def mesh_from_labels(labels, step_size=1, min_voxels=8, spacing=(1.0, 1.0, 1.0),
                     origin=(0.0, 0.0, 0.0), clip_z=None, smooth=0.0, ids=None):
    """Per-instance marching cubes over an int label volume, concatenated into ONE mesh.

    Returns ``(verts_plot, faces, normals, verts_idx, owner)`` where `owner` is the label id of
    each triangle -- the hook for per-cell colour.

    One march per instance on its own padded bounding box, reusing the `ndi.find_objects` idiom
    from `scene.py`. Marching a single binary `labels > 0` instead would FUSE touching cells:
    in confluent tissue neighbouring territories share a face, so the result is one connected
    blob with no cell boundaries at all, and per-cell colour becomes impossible.

    Everything ends up in a single `Mesh3d`; one trace per cell would stall the browser at a few
    hundred cells.
    """
    lab = np.asarray(labels)
    objs = ndi.find_objects(lab)
    parts = []
    for n, sl in enumerate(objs, start=1):
        if sl is None or (ids is not None and n not in ids):
            continue
        sub = lab[sl] == n
        if sub.sum() < min_voxels:
            continue
        v, f, nrm, vi = mesh_from_field(sub.astype(np.float32), 0.5, "high", step_size,
                                        spacing=(1.0, 1.0, 1.0), pad=True, smooth=smooth)
        if v is None:
            continue
        vi = vi + np.array([s.start for s in sl], np.float32)     # local -> global index frame
        parts.append((vi, f, nrm, n))
    if not parts:
        return None, None, None, None, None

    offs = np.cumsum([0] + [len(p[0]) for p in parts])[:-1]
    verts_idx = np.concatenate([p[0] for p in parts])
    faces = np.concatenate([p[1] + o for p, o in zip(parts, offs)])
    normals = np.concatenate([p[2] for p in parts])
    owner = np.repeat([p[3] for p in parts], [len(p[1]) for p in parts]).astype(np.int32)

    verts_plot = verts_idx * np.asarray(spacing, np.float32) + np.asarray(origin, np.float32)
    if clip_z is not None:
        verts_plot[:, 0] = np.clip(verts_plot[:, 0], clip_z[0], clip_z[1])
    return (verts_plot.astype(np.float32), faces.astype(np.int32), normals,
            verts_idx.astype(np.float32), owner)


def mesh_volume(verts_plot, faces):
    """Signed volume of a closed triangle mesh, via the divergence theorem.

    Used as a correctness check: |V| should match the voxel count times the voxel volume, and a
    NEGATIVE value means the triangle winding is inverted -- which is the thing that makes flat
    shading light the surface from the inside. Cheaper and far more reliable than eyeballing it.
    """
    t = verts_plot[faces]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


# --------------------------------------------------------------------- colour

def hex_from_rgb8(rgb8):
    """(N, 3) uint8 -> ["#RRGGBB", ...], vectorised.

    plotly serialises numeric arrays as binary but a list of strings as plain JSON, so this is
    the expensive part of a large mesh (~11 bytes/face). It is still the right choice: the
    reference viewer proves plotly.js accepts CSS strings here, whereas an (N, 3) numeric
    `facecolor` is only known to pass plotly.py's validator.
    """
    a = np.ascontiguousarray(rgb8, np.uint8)
    h = np.frombuffer(a.tobytes().hex().upper().encode(), "S6").astype("U6")
    return np.char.add("#", h).tolist()


def sample_face_colours(verts_idx, faces, normals, rgb, mask=None, inset_vox=1.0, order=1,
                        gamma=1.0, floor=0.0):
    """Marker RGB sampled at each triangle -> list of "#RRGGBB", one per face.

    The sample point is the triangle centroid stepped `inset_vox` voxels INWARD along the
    averaged vertex normal. That inset matters more than the interpolation order:
    `render_marker_3d` multiplies every marker by `boundary_falloff(d)`, so a sample taken
    exactly ON the surface sits in the falloff skirt and is systematically dim -- and because
    `to_rgb` then percentile-scales each channel, the whole mesh comes out washed out and nearly
    uniform.

    The inward direction is verified against `mask` rather than assumed: skimage's normal
    orientation depends on `gradient_direction`, and silently sampling outward would give a
    uniformly black mesh. If the majority of inset points land outside the mask, the sign flips.

    The trade-off is real and worth knowing: a MEMBRANE marker peaks at tau = 1, i.e. exactly on
    the surface, so a large inset walks off its peak (measured: a membrane marker dims by ~85%
    at inset_vox=3). An INTERIOR marker does the opposite. 1.0 is a compromise -- enough to clear
    the falloff skirt without leaving a membrane band. Set 0 to sample the surface literally.
    """
    cen = verts_idx[faces].mean(axis=1)
    if inset_vox and normals is not None:
        nrm = normals[faces].mean(axis=1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        p = cen - inset_vox * nrm
        if mask is not None:
            q = cen + inset_vox * nrm
            inside_p = _fraction_inside(p, mask)
            inside_q = _fraction_inside(q, mask)
            if inside_q > inside_p:
                p = q
    else:
        p = cen

    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.rint(np.asarray(rgb, np.float32) * 255.0), 0, 255).astype(np.uint8)
    vals = np.stack([ndi.map_coordinates(rgb[..., c].astype(np.float32), p.T,
                                         order=order, mode="nearest")
                     for c in range(rgb.shape[-1])], -1)
    vals = np.clip(vals / 255.0, 0.0, 1.0)
    if gamma != 1.0:
        vals = vals ** float(gamma)
    if floor:
        vals = floor + (1.0 - floor) * vals
    return hex_from_rgb8(np.clip(np.rint(vals * 255.0), 0, 255).astype(np.uint8))


def _fraction_inside(p, mask):
    idx = np.clip(np.rint(p).astype(int), 0, np.array(mask.shape) - 1)
    return float(mask[idx[:, 0], idx[:, 1], idx[:, 2]].mean())


def instance_colours(owner, colours=None):
    """Per-face colour from the per-face instance id (the reference viewer's scheme)."""
    if colours is None:
        colours = PLOTLY_COLORWAY
    if isinstance(colours, dict):
        return [colours.get(int(o), PLOTLY_COLORWAY[0]) for o in owner]
    lut = np.asarray(colours, dtype=object)
    return lut[(owner - 1) % len(lut)].tolist()


# --------------------------------------------------------------------- traces

def _orient(verts, faces, flip_winding="auto"):
    """Wind the triangles so the surface normals point OUTWARD in the plotly (x, y, z) frame.

    Two sign flips fight each other here and hardcoding either one is a trap:

    * mapping (z, y, x) -> (x, y, z) is an ODD permutation, so it reverses handedness;
    * skimage's winding already depends on `gradient_direction`, and its raw output happens to
      be negative under a (z, y, x) reading -- so the permutation puts it right rather than
      wrong.

    Rather than assume, measure: the divergence-theorem volume of a closed mesh is positive
    exactly when the winding is outward. `"auto"` computes it in the frame we are about to plot
    in and flips only if needed, which also stays correct if skimage ever changes convention.
    Pass True/False to force it.
    """
    xyz = verts[:, ::-1]                                   # (z,y,x) -> (x,y,z)
    if flip_winding == "auto":
        flip_winding = mesh_volume(xyz, faces) < 0
    return faces[:, ::-1] if flip_winding else faces


def _mesh3d(verts, faces, *, flip_winding="auto", **kw):
    go = _go()
    f = _orient(verts, faces, flip_winding)
    return go.Mesh3d(x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
                     i=np.ascontiguousarray(f[:, 0]), j=np.ascontiguousarray(f[:, 1]),
                     k=np.ascontiguousarray(f[:, 2]), **kw)


def _decimate_for(n_faces, step_size, max_faces, what):
    """How much to coarsen so a mesh fits under `max_faces`. Faces scale as 1/step**2."""
    if n_faces <= max_faces:
        return step_size
    new = int(np.ceil(np.sqrt(n_faces / max_faces) * step_size))
    warnings.warn(f"{what}: {n_faces:,} faces exceeds max_faces={max_faces:,}; "
                  f"re-meshing at step_size={new} (was {step_size}). "
                  f"Pass auto_decimate=False to keep the full mesh.", stacklevel=3)
    return new


def _mesh_edges(faces):
    """Unique undirected edges of a triangle mesh, as an (E, 2) int array."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def wireframe_trace(verts, faces, colour="#111820", width=1.2, opacity=0.5, name="outline"):
    """A mesh's own triangulation as ONE `go.Scatter3d` polyline, edges broken by NaN.

    Cheap and, for a per-instance mesh, exactly right for "outline the cells": `mesh_from_labels`
    marches each label on its own padded bounding box (see its own docstring), so two touching
    cells never share a vertex -- every edge already sits entirely on one cell's own surface, and
    a plain wireframe of the combined mesh reads as a per-cell outline rather than a fused blob's
    interior grid. NaN (not None) breaks the line between edges: plotly.js treats a NaN exactly
    like a gap, and keeping everything numeric lets this build with vectorised numpy instead of
    a million-element Python list.
    """
    go = _go()
    e = _mesh_edges(np.asarray(faces))
    xyz = np.asarray(verts)[:, ::-1]                            # (z, y, x) -> (x, y, z)
    p = xyz[e]                                                  # (E, 2, 3)
    n = len(p)
    out = np.full((3, 3 * n), np.nan, np.float32)
    for a in range(3):
        out[a, 0::3] = p[:, 0, a]
        out[a, 1::3] = p[:, 1, a]
    xs, ys, zs = out
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color=colour, width=width),
                        opacity=opacity, name=name, showlegend=False, hoverinfo="skip")


def surface_traces(cell=None, nuc=None, *, d=None, tau=None, labels=None, nuc_labels=None,
                   rgb=None, colour_by=None, colours=None,
                   spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), clip_z=None,
                   step_size=1, nuc_step_size=None, smooth=0.0,
                   colour="#AB63FA", nuc_colour=_NUC_COLOUR,
                   opacity=0.55, nuc_opacity=1.0, flatshading=True, flip_winding="auto",
                   inset_vox=1.0, sample_order=1, gamma=1.0,
                   max_faces=250_000, auto_decimate=True, hover=False,
                   name="cells", nuc_name="nuclei", show_nucleus=True,
                   edges=False, edge_colour="#111820", edge_width=1.2, edge_opacity=0.5,
                   edge_name=None):
    """Cell and nucleus surfaces as `go.Mesh3d`. Returns a list of 0-3 traces.

    Field preference, best first: `d` (analytic, sub-voxel) > `cell` (boolean mask) > `labels`
    (multi-instance). Likewise `tau` > `nuc` > `nuc_labels` for the nucleus.

    `edges=True` adds one extra `go.Scatter3d` wireframe trace over the CELL mesh (not the
    nucleus): with `colour_by="instance"`, same-type neighbours are the same fill colour, so at
    tissue scale the outline is what actually separates one cell from the next. Reuses the exact
    mesh already built for the fill (see `wireframe_trace`), so it costs no extra marching cubes.
    """
    traces, info = [], {}
    mesh_kw = dict(spacing=spacing, origin=origin, clip_z=clip_z, smooth=smooth)

    # ---- cell ----------------------------------------------------------------
    owner = None
    for attempt in range(2):
        if labels is not None:
            v, f, nrm, vi, owner = mesh_from_labels(labels, step_size, spacing=spacing,
                                                    origin=origin, clip_z=clip_z, smooth=smooth)
        elif d is not None:
            v, f, nrm, vi = mesh_from_field(d, 1.0, "low", step_size, **mesh_kw)
        elif cell is not None:
            v, f, nrm, vi = mesh_from_field(np.asarray(cell, np.float32), 0.5, "high",
                                            step_size, **mesh_kw)
        else:
            v = f = nrm = vi = None
        if v is None:
            break
        if auto_decimate and attempt == 0:
            new = _decimate_for(len(f), step_size, max_faces, name)
            if new != step_size:
                step_size = new
                continue
        break

    if v is not None:
        mask = cell if cell is not None else (labels > 0 if labels is not None else None)
        mode = colour_by
        if mode is None:
            mode = "rgb" if rgb is not None else ("instance" if owner is not None else "flat")
        kw = dict(opacity=opacity, flatshading=flatshading, showscale=False, name=name,
                  hoverinfo="all" if hover else "skip", showlegend=True)
        if mode == "rgb" and rgb is not None:
            kw["facecolor"] = sample_face_colours(vi, f, nrm, rgb, mask, inset_vox,
                                                  sample_order, gamma)
        elif mode == "instance" and owner is not None:
            kw["facecolor"] = instance_colours(owner, colours)
        else:
            kw["color"] = colour
        traces.append(_mesh3d(v, f, flip_winding=flip_winding, **kw))
        info["cell_faces"] = len(f)
        info["cell_step"] = step_size
        if edges:
            traces.append(wireframe_trace(v, f, colour=edge_colour, width=edge_width,
                                          opacity=edge_opacity,
                                          name=edge_name or f"{name} outline"))
            info["edge_count"] = int(len(_mesh_edges(f)))

    # ---- nucleus -------------------------------------------------------------
    if show_nucleus:
        ns = nuc_step_size or step_size
        if nuc_labels is not None:
            nv, nf, _, _, nown = mesh_from_labels(nuc_labels, ns, spacing=spacing,
                                                  origin=origin, clip_z=clip_z, smooth=smooth)
        elif tau is not None:
            # tau is only meaningful inside the cell; push the far field positive so no stray
            # level-0 crossings appear outside, then march the tau = 0 envelope.
            t = np.where(np.asarray(cell, bool), np.asarray(tau, np.float32), 1.0) \
                if cell is not None else np.asarray(tau, np.float32)
            nv, nf, _, _ = mesh_from_field(t, 0.0, "low", ns, **mesh_kw)
            nown = None
        elif nuc is not None:
            nv, nf, _, _ = mesh_from_field(np.asarray(nuc, np.float32), 0.5, "high", ns,
                                           **mesh_kw)
            nown = None
        else:
            nv = nf = nown = None
        if nv is not None:
            kw = dict(opacity=nuc_opacity, flatshading=flatshading, showscale=False,
                      name=nuc_name, hoverinfo="all" if hover else "skip", showlegend=True)
            if nown is not None and colour_by == "instance":
                kw["facecolor"] = instance_colours(nown, colours)
            else:
                kw["color"] = nuc_colour
            traces.append(_mesh3d(nv, nf, flip_winding=flip_winding, **kw))
            info["nuc_faces"] = len(nf)

    return traces, info


def _block_mean(vol, factor):
    """Downsample by integer factors with a block MEAN (never striding).

    Striding aliases exactly the fine textures this viewer exists to show; averaging degrades
    them gracefully instead.
    """
    fz, fy, fx = factor
    if (fz, fy, fx) == (1, 1, 1):
        return vol, (0, 0, 0)
    nz, ny, nx = (s // f * f for s, f in zip(vol.shape, factor))
    v = vol[:nz, :ny, :nx]
    return v.reshape(nz // fz, fz, ny // fy, fy, nx // fx, fx).mean((1, 3, 5)), (0, 0, 0)


def volume_traces(channels, mask=None, *, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
                  channel_names=None, channel_colours=("red", "green", "blue"),
                  iso_pct=(25.0, 99.5), iso_floor_frac=1e-3, transform="log", log_decades=2.5, opacity=0.16,
                  opacityscale=((0.0, 0.0), (0.25, 0.05), (0.6, 0.35), (1.0, 1.0)),
                  surface_count=17, max_voxels=64 ** 3, crop_to_mask=True, crop_pad=3,
                  downsample=None, composite=False, show_caps=False, lighting=None):
    """Raymarched `go.Volume`, one trace per marker channel. Returns (traces, info).

    Three overlaid single-hue traces rather than one composite: it keeps channel identity and
    gives a legend toggle per marker, which is the single most useful interaction here.
    `go.Volume` has no RGB mode, so `composite=True` (one scalar, magma) is the 3x smaller
    alternative when file size matters more than identity.

    Size is the binding constraint: roughly 21 bytes per voxel per trace, so 64**3 is ~5.6 MB
    per channel and 128**3 is ~45 MB. The default reduction is (a) crop to the mask bounding box
    -- a single cell typically fills only a few percent of its box, so this is nearly free --
    then (b) block-mean downsample until the grid fits.
    """
    go = _go()
    chans = [np.asarray(c, np.float32) for c in channels]
    shape = chans[0].shape
    m = np.asarray(mask, bool) if mask is not None else np.ones(shape, bool)
    off = np.zeros(3, np.float32)

    if crop_to_mask and m.any():
        sl = ndi.find_objects(m.astype(np.uint8))[0]
        sl = tuple(slice(max(0, s.start - crop_pad), min(n, s.stop + crop_pad))
                   for s, n in zip(sl, shape))
        chans = [c[sl] for c in chans]
        m = m[sl]
        off = np.array([s.start for s in sl], np.float32)

    fac = np.ones(3, int) if downsample is None else np.asarray(downsample, int)
    if downsample is None:
        while np.prod([s // f for s, f in zip(chans[0].shape, fac)]) > max_voxels:
            fac = fac + 1
    if (fac != 1).any():
        chans = [_block_mean(c, tuple(fac))[0] for c in chans]
        m = _block_mean(m.astype(np.float32), tuple(fac))[0] > 0.5

    sp = np.asarray(spacing, np.float32) * fac
    org = np.asarray(origin, np.float32) + off * np.asarray(spacing, np.float32)
    nz, ny, nx = chans[0].shape

    # plotly wants flat x/y/z/value of equal length. Transposing to (x, y, z) first and using
    # meshgrid(indexing="ij") reproduces the layout every documented example uses, which removes
    # any doubt about how plotly.js reconstructs the lattice.
    zc = org[0] + np.arange(nz, dtype=np.float32) * sp[0]
    yc = org[1] + np.arange(ny, dtype=np.float32) * sp[1]
    xc = org[2] + np.arange(nx, dtype=np.float32) * sp[2]
    Xg, Yg, Zg = np.meshgrid(xc, yc, zc, indexing="ij")
    gx, gy, gz = (Xg.ravel().astype(np.float32), Yg.ravel().astype(np.float32),
                  Zg.ravel().astype(np.float32))

    if composite:
        chans = [np.max(np.stack(chans), 0)]
        channel_colours = ("magma",)
        channel_names = (channel_names[0] if channel_names else "markers",)

    names = channel_names or [f"marker {i}" for i in range(len(chans))]
    lighting = lighting or dict(ambient=0.9, diffuse=0.3, specular=0.05)
    traces = []
    for i, ch in enumerate(chans):
        vals = ch[m]
        if vals.size == 0:
            continue
        # Percentiles over the POSITIVE values, not all of them. A localised marker (nuclear,
        # punctate) is exactly zero over most of the cell, so an all-values percentile collapses
        # to 0, isomin goes to 0, and the trace draws every empty voxel as fog.
        pos = vals[vals > 0]
        pos = pos if pos.size else vals

        if transform == "log":
            # `render_marker_3d` builds texture as exp(strength * field), so marker intensity is
            # LOGNORMAL: measured skew 3.7 raw vs -0.5 in log. On a linear ramp almost the whole
            # cell is squashed into the bottom few percent and only the brightest peaks survive,
            # which is why filaments and diffuse texture vanish. Taking the log first spreads the
            # actual distribution across the colour and opacity ramps.
            # eps sets the noise floor `log_decades` below the bright end. Tying it to the
            # peak rather than to a low percentile matters: a localised marker runs all the way
            # down to 0, so a percentile-based eps collapses to ~1e-6 and isomin lands at
            # log(1e-6) = -13.8, putting the whole interesting range in the top 5% of the ramp.
            top = float(np.percentile(pos, iso_pct[1]))
            eps = max(top * 10.0 ** (-float(log_decades)), 1e-12)
            lo, hi = np.percentile(np.log(pos + eps), iso_pct)
            lo = max(lo, np.log(2.0 * eps))
            v = np.where(m, np.log(np.maximum(ch, 0.0) + eps), lo - 1.0)
        else:
            lo, hi = np.percentile(pos, iso_pct)
            # a marker with huge dynamic range still lands its low percentile in the near-zero
            # regime, so floor isomin at a small fraction of isomax
            lo = max(float(lo), float(hi) * iso_floor_frac)
            v = np.where(m, ch, 0.0)

        if not np.isfinite([lo, hi]).all() or hi <= lo:
            warnings.warn(f"channel {names[i]!r} has no dynamic range in the mask; skipped.",
                          stacklevel=2)
            continue
        key = channel_colours[i % len(channel_colours)]
        traces.append(go.Volume(
            x=gx, y=gy, z=gz,
            value=np.ascontiguousarray(np.transpose(v, (2, 1, 0)), np.float32).ravel(),
            isomin=float(lo), isomax=float(hi), opacity=float(opacity),
            opacityscale=[list(p) for p in opacityscale],
            surface_count=int(surface_count),
            colorscale=CHANNEL_SCALES.get(key, key), showscale=False,
            caps_x_show=show_caps, caps_y_show=show_caps, caps_z_show=show_caps,
            lighting=lighting, name=names[i], showlegend=True, hoverinfo="skip"))
    return traces, dict(volume_grid=(nz, ny, nx), downsample=tuple(int(x) for x in fac))


def focal_plane_trace(xlim, ylim, z, colour=_FOCAL_COLOUR, opacity=0.18, name="focal plane",
                      overhang=0.05):
    go = _go()
    (x0, x1), (y0, y1) = xlim, ylim
    px, py = (x1 - x0) * overhang, (y1 - y0) * overhang
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
    return go.Mesh3d(x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], z=[z] * 4,
                     i=[0, 0], j=[1, 2], k=[2, 3], color=colour, opacity=opacity,
                     name=name, showlegend=True, hoverinfo="skip", showscale=False)


def section_box_trace(xlim, ylim, z0, z1, colour=_BOX_COLOUR, width=2, name="section box"):
    """Wireframe of the physical section: 12 edges as one polyline broken by None."""
    go = _go()
    (x0, x1), (y0, y1) = xlim, ylim
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    xs, ys, zs = [], [], []
    for z in (z0, z1):
        for a in range(4):
            b = (a + 1) % 4
            xs += [ring[a][0], ring[b][0], None]
            ys += [ring[a][1], ring[b][1], None]
            zs += [z, z, None]
    for a in range(4):
        xs += [ring[a][0], ring[a][0], None]
        ys += [ring[a][1], ring[a][1], None]
        zs += [z0, z1, None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color=colour, width=width),
                        name=name, showlegend=True, hoverinfo="skip")


# --------------------------------------------------------------------- entry point

def _data_extent(traces, pad=0.08):
    """Lateral (x, y) range actually occupied by the geometry, padded. (None, None) if empty."""
    xs = [np.asarray(t.x, float) for t in traces
          if getattr(t, "x", None) is not None and t.type in ("mesh3d", "volume")]
    ys = [np.asarray(t.y, float) for t in traces
          if getattr(t, "y", None) is not None and t.type in ("mesh3d", "volume")]
    if not xs or not ys:
        return None, None
    x, y = np.concatenate(xs), np.concatenate(ys)
    (x0, x1), (y0, y1) = (x.min(), x.max()), (y.min(), y.max())
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return (float(x0 - px), float(x1 + px)), (float(y0 - py), float(y1 + py))


def _first_shape(*arrays):
    for a in arrays:
        if a is not None:
            a = np.asarray(a)
            return a.shape[:3] if a.ndim == 4 else a.shape
    return None


def scene3d(cell=None, nuc=None, *, mode="surface",
            d=None, tau=None, labels=None, nuc_labels=None,
            channels=None, rgb=None, mask=None,
            opt=None, units=None, shape=None,
            section_um=None, section_centre_um=0.0, clip_to_section=False,
            show_focal_plane=True, show_section_box=True, overlay_extent="data",
            title=None, camera_eye=DEFAULT_CAMERA, showlegend=True,
            width=900, height=750, template="plotly_white", shell=None,
            step_size=1, **kwargs):
    """Interactive 3D view of a synthetic cell or tissue. Returns a `plotly.graph_objects.Figure`.

    mode="surface"
        Marching-cubes surfaces of the cell and nucleus, optionally coloured per triangle by the
        marker RGB (`rgb=`) or per instance by phenotype (`labels=` + `colours=`). Reproduces
        the previous project's viewer, and extends unchanged to a tissue label volume.
    mode="volume"
        Raymarched `go.Volume` per marker channel (`channels=`), so the interior 3D texture is
        visible, plus an optional translucent cell shell.

    Physical frame: pass `opt` (an `optics.Optics`) and everything is in microns with the volume
    centred on z = 0, so `opt.focal_um` and a section at `section_centre_um` are directly
    plottable. Without it, units are voxels.

    Does NOT call `.show()` -- like `plot3d`, it hands the figure back so the caller decides.
    """
    go = _go()
    shape = shape or _first_shape(cell, d, tau, nuc, labels, nuc_labels, mask, rgb,
                                  channels[0] if channels else None)
    if shape is None:
        raise ValueError("scene3d: pass at least one volume (cell, d, labels, channels, ...)")
    nz, ny, nx = shape

    units = units or ("um" if opt is not None else "px")
    if units == "um":
        if opt is None:
            raise ValueError("units='um' needs an Optics instance (opt=)")
        spacing = np.array([opt.um_per_pz, opt.um_per_px, opt.um_per_px], np.float32)
        origin = -((np.array(shape, np.float32) - 1.0) / 2.0) * spacing
        unit = "µm"
    else:
        spacing = np.ones(3, np.float32)
        origin = np.zeros(3, np.float32)
        unit = "px"

    zlim = (float(origin[0]), float(origin[0] + (nz - 1) * spacing[0]))
    ylim = (float(origin[1]), float(origin[1] + (ny - 1) * spacing[1]))
    xlim = (float(origin[2]), float(origin[2] + (nx - 1) * spacing[2]))

    # ---- section geometry ----------------------------------------------------
    z_lo = z_hi = None
    if section_um:
        half = section_um / 2.0
        if units == "um":
            z_lo, z_hi = section_centre_um - half, section_centre_um + half
        else:
            c = section_centre_um / opt.um_per_pz + (nz - 1) / 2.0 if opt else (nz - 1) / 2.0
            h = half / opt.um_per_pz if opt else half
            z_lo, z_hi = c - h, c + h
        if z_hi < zlim[0] or z_lo > zlim[1]:
            warnings.warn("section lies entirely outside the volume; not drawn.", stacklevel=2)
            z_lo = z_hi = None
        else:
            z_lo, z_hi = max(z_lo, zlim[0]), min(z_hi, zlim[1])
    clip_z = (z_lo, z_hi) if (clip_to_section and z_lo is not None) else None

    common = dict(spacing=tuple(spacing), origin=tuple(origin), clip_z=clip_z)
    traces, info = [], {}

    if mode == "surface":
        if cell is None and d is None and labels is None:
            raise ValueError("mode='surface' needs cell=, d= or labels=")
        t, i = surface_traces(cell=cell, nuc=nuc, d=d, tau=tau, labels=labels,
                              nuc_labels=nuc_labels, rgb=rgb, step_size=step_size,
                              **common, **kwargs)
        traces += t
        info.update(i)
    elif mode == "volume":
        chans = channels
        if chans is None and rgb is not None:
            chans = [np.asarray(rgb)[..., c] for c in range(np.asarray(rgb).shape[-1])]
        if chans is None:
            raise ValueError("mode='volume' needs channels= (or rgb=)")
        m = mask if mask is not None else (cell if cell is not None else
                                           (labels > 0 if labels is not None else None))
        vkw = {k: kwargs.pop(k) for k in list(kwargs)
               if k in ("channel_names", "channel_colours", "iso_pct", "iso_floor_frac",
                        "transform", "log_decades", "opacity",
                        "opacityscale", "surface_count", "max_voxels", "crop_to_mask",
                        "crop_pad", "downsample", "composite", "show_caps", "lighting")}
        t, i = volume_traces(chans, m, spacing=tuple(spacing), origin=tuple(origin), **vkw)
        traces += t
        info.update(i)
        # the shell goes LAST: plotly.js composites transparent gl3d objects in trace order
        if (shell if shell is not None else (cell is not None or labels is not None)):
            st, si = surface_traces(cell=cell, d=d, labels=labels, colour_by="flat",
                                    colour="#aab4c2", opacity=0.06, flatshading=False,
                                    show_nucleus=False, step_size=step_size + 1,
                                    name="cell surface", **common, **kwargs)
            traces += st
            info.update(si)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'surface' or 'volume'")

    # Overlays are sized to the GEOMETRY, not the volume. A sandbox box is mostly empty air
    # around one cell, and a focal plane spanning all of it turns the cell into a speck and
    # flattens the whole scene under aspectmode="data". In a real tissue tile the two extents
    # coincide anyway, so this costs nothing there.
    oxlim, oylim = (_data_extent(traces) if overlay_extent == "data" else (xlim, ylim))
    oxlim, oylim = oxlim or xlim, oylim or ylim

    if opt is not None and show_focal_plane:
        zf = opt.focal_um if units == "um" else opt.focal_um / opt.um_per_pz + (nz - 1) / 2.0
        if zlim[0] <= zf <= zlim[1]:
            traces.append(focal_plane_trace(oxlim, oylim, zf))
    if z_lo is not None and show_section_box:
        traces.append(section_box_trace(oxlim, oylim, z_lo, z_hi))

    if title is None:
        title = _auto_title(labels, nuc_labels, section_um, mode, info, unit)

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0), title=title, showlegend=showlegend,
        width=width, height=height, template=template,
        scene=dict(xaxis_title=f"col ({unit})", yaxis_title=f"row ({unit})",
                   zaxis_title=f"depth z ({unit})", aspectmode="data",
                   camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]))))
    fig._viewer_info = info                     # handy for tests / notebook printouts
    return fig


def _auto_title(labels, nuc_labels, section_um, mode, info, unit):
    if labels is not None:
        n = int(np.unique(labels).size - (1 if (labels == 0).any() else 0))
        bits = [f"{n} cell{'s' if n != 1 else ''}"]
        if nuc_labels is not None:
            k = int(np.unique(nuc_labels).size - (1 if (nuc_labels == 0).any() else 0))
            bits.append(f"({k} with a nucleus)")
    else:
        bits = ["1 cell"]
    if section_um:
        bits.insert(1, f"in a {section_um:g} µm section")
    if mode == "volume" and "volume_grid" in info:
        bits.append("— volume {}×{}×{}".format(*info["volume_grid"]))
    elif "cell_faces" in info:
        bits.append(f"— {info['cell_faces']:,} faces")
    return " ".join(bits)


def save(fig, path, include_plotlyjs="cdn", full_html=True, config=None, div_id=None,
         auto_open=False, verbose=True):
    """Write a standalone HTML file.

    `include_plotlyjs="cdn"` (the default) keeps the file ~4.8 MB smaller but needs a network
    connection to view; pass True for a genuinely offline, self-contained page.
    """
    from pathlib import Path
    import plotly.io as pio
    p = Path(path)
    pio.write_html(fig, file=str(p), include_plotlyjs=include_plotlyjs, full_html=full_html,
                   config=config, div_id=div_id, auto_open=auto_open)
    if verbose:
        print(f"wrote {p}  ({p.stat().st_size/1e6:.2f} MB, plotly.js={include_plotlyjs!r})")
    return p
