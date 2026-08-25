import numpy as np
import dataclasses
from scipy import ndimage as ndi

from scene import support_mask, thin, vox_to_world, body_frame, make_boundary, patch_grid, centred_grid_3d, TapeDict
from render import render_marker
import config as cfg
from parameter import CellGeometry, FlatNoise

def artifact_sites(tape, AR, shape, n, tissue_support, skip = 0):
    nz, ny, nx = shape
    sup = support_mask(tape["art_support3"], AR.SUPPORT_SCALE.v, AR.SUPPORT_SCALE_Z.v,
                       AR.COVER.v)
    if tissue_support is not None:
        sup = sup & np.asarray(tissue_support, bool)
    col = sup.any(0)

    xy = tape["art_xy"]
    keep = thin(xy, tape["art_order"], AR.MIN_DIST.v)
    cx, cy = xy[keep, 0], xy[keep, 1]
    keep = keep[col[np.clip(cy.astype(int), 0, ny - 1), 
                    np.clip(cx.astype(int), 0, nx - 1)]]

    take = keep[skip:skip + max(int(round(n)), 0)]
    return xy[take], take

def marker_affinity(panel, names, dapi_name=cfg.DAPI_NAME):
    """Which markers artifacts contaminate, and how strongly. DAPI is always exactly 0."""
    out = {nm: float(panel.Markers[nm].artifact_affinity.v) for nm in names}
    out[dapi_name] = 0.0
    return out

def _perimeter_point(s, ny, nx, pad):
    """s in [0,1) -> a point on the frame boundary, pushed `pad` voxels OUTSIDE it.
    """
    H, W = ny + 2.0 * pad, nx + 2.0 * pad
    t = (float(s) % 1.0) * 2.0 * (H + W)
    if t < W:
        # top
        return (-pad, t - pad)
    t -= W
    if t < H:
        # right
        return (t - pad, nx + pad)
    t -= H
    if t < W:
        # bottom
        return (ny + pad, nx + pad - t)
    t -= W
    # left
    return (ny + pad - t, -pad)

def _mode_envelope(n_mode, beta, target=2.0):
    """m^-beta, renormalised so the wobble's RMS is 1 whatever beta or n_mode are.
    """
    m = np.arange(1, n_mode + 1, dtype=float)
    w = m ** -float(beta)
    return w * np.sqrt(target / max(float((w ** 2).sum()), 1e-30))

def _slab_bounds(shape, opt, margin_vox=1.0):
    """The voxel range kryostat will actually keep, minus a margin."""
    nz = shape[0]
    c = (nz - 1) / 2.0
    lim = 0.5 * float(opt.section_um) / float(opt.um_per_pz)
    lo, hi = c - lim + margin_vox, c + lim - margin_vox
    # An object thicker than the section cannot fit inside it at all; centre it and let the cut
    # take the top and bottom off symmetrically
    return (c, c) if lo > hi else (lo, hi)


def _z_centre_vox(z_frac, shape, opt, margin_vox=1.0):
    """Axial centre in VOXEL index, from a signed fraction of the half-section.
    """
    nz = shape[0]
    z_um = -float(z_frac) * 0.5 * float(opt.section_um)
    z_vox = (nz - 1) / 2.0 + z_um / float(opt.um_per_pz)
    lo, hi = _slab_bounds(shape, opt, margin_vox)
    return float(np.clip(z_vox, lo, hi))


def fibre_centreline(tape, j, F, shape, opt, um_per_vox, n_t=512, pad=None, width_vox=1.0):
    """The fibre's path: a chord across the frame, bent by a few frozen modes.
    Returns pts (n_t, 3) in (z, y, x) voxel coordinates.
    """
    nz, ny, nx = shape
    pad = float(np.ceil(2.0 * width_vox + 4.0)) if pad is None else pad
    sA = tape["u_fib_end"][j, 0]
    # Move the exit point sB between 0.3 - 0.7 of a full lap further 
    sB = sA + 0.30 + 0.40 * tape["u_fib_end"][j, 1]
    
    # Compute start and endpoint on the perimeter
    ay, ax = _perimeter_point(sA, ny, nx, pad)
    by, bx = _perimeter_point(sB, ny, nx, pad)

    # Compute the straight connecting line between the points
    dy, dx = by - ay, bx - ax
    norm = np.hypot(dy, dx) + 1e-12
    # in-plane normal to the chord
    ny_hat, nx_hat = -dx / norm, dy / norm            

    # Which fraction of the chord is actually drawn.
    # Threshould drawn value with learnable param
    u = tape["u_fib_span"][j]
    p_in, trim = float(F.P_END_INSIDE.v), float(F.END_TRIM.v)
    t_min = min(1.6 * pad / norm, 0.5 * trim)
    t0 = t_min + (trim - t_min) * u[2] if u[0] < p_in else 0.0
    t1 = 1.0 - (t_min + (trim - t_min) * u[3]) if u[1] < p_in else 1.0
    t = np.linspace(t0, max(t1, t0 + 0.05), int(n_t))
    yy = ay + t * (by - ay)
    xx = ax + t * (bx - ax)

    # 1D version of the same spectral trick in make_boundary to generate unit RMS
    n_mode = tape["z_fib_lat"].shape[1]
    w = _mode_envelope(n_mode, F.BETA.v)
    basis = np.sin(np.pi * np.arange(1, n_mode + 1)[None, :] * t[:, None])
    
    lat = (basis * (w * tape["z_fib_lat"][j, :n_mode])).sum(1)
    lat_vox = float(F.WOBBLE_UM.v) / float(um_per_vox)
    yy = yy + lat_vox * lat * ny_hat
    xx = xx + lat_vox * lat * nx_hat

    # Add z-Axis wobble, but make sure it stays below cover slip
    z0 = _z_centre_vox(F.Z_FRAC.v, shape, opt, margin_vox=width_vox + 1.0)
    zlo, zhi = _slab_bounds(shape, opt, margin_vox=width_vox + 1.0)
    room = max(min(z0 - zlo, zhi - z0), 0.0)
    ax_raw = (basis * (w * tape["z_fib_ax"][j, :n_mode])).sum(1)
    zz = z0 + float(F.Z_WOBBLE_FRAC.v) * np.tanh(ax_raw) * room
    return np.stack([zz, yy, xx], -1)

def make_fussel(tape, F, j, shape, opt, cfg, um_per_vox, n_t, spacing):
    nz, ny, nx = shape
    # get radius of fussel
    r_um = 0.5 * float(F.WIDTH_UM.v) * np.exp(float(F.WIDTH_SIGMA.v) * tape["z_art_size"][j])
    # prevent R going -> 0
    R = max(float(r_um) / float(um_per_vox), 0.6)
    
    pts = fibre_centreline(tape, j, F, shape, opt, um_per_vox, n_t=n_t, width_vox=R)
    
    # Crop in z before the distance transform.
    zl = int(max(0, np.floor(pts[:, 0].min() - 2.0 * R - 2)))
    zh = int(min(nz, np.ceil(pts[:, 0].max() + 2.0 * R + 3)))
    sub = (slice(zl, zh), slice(0, ny), slice(0, nx))
    sz = (zh - zl, ny, nx)
    
    # rasterise densely enough to stay connected
    length = float(np.sum(np.linalg.norm(np.diff(pts[:, 1:], axis=0), axis=1)))
    n_s = max(int(np.ceil(2.5 * length)), n_t)
    ti = np.linspace(0.0, 1.0, n_s)
    ps = np.stack(
        [np.interp(ti, np.linspace(0, 1, n_t), pts[:, k]) for k in range(3)], -1
    )
    iz = np.rint(ps[:, 0]).astype(int) - zl
    iy = np.rint(ps[:, 1]).astype(int)
    ix = np.rint(ps[:, 2]).astype(int)
    ok = (iz >= 0) & (iz < sz[0]) & (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    if not ok.any():
        # the whole fibre missed the volume -- legal (it can wander outside), just render nothing
        return None
    
    # initialize seed map with false for the fussel voxels for distance transform
    seed = np.ones(sz, bool)
    seed[iz[ok], iy[ok], ix[ok]] = False
    # Returns the distance to nearest centreline and the nearest pixel on centreline
    dist, inds = ndi.distance_transform_edt(seed, sampling=spacing, return_indices=True)
    mask = dist <= R
    # get normalised radial coordinate: 0 on the centreline, 1 exactly at the surface,
    # clip due to large tau size of tile spanning fibres
    d = np.minimum(dist / R, cfg._TAU_CLIP)
    
    # Generate PHI
    # Initialize plain voxel index meshgrid
    zz, yy, xx = np.meshgrid(
        *[np.arange(s, dtype=np.float32) for s in sz], indexing="ij"
    )
    # Compute displacement from nearest centreline by subtracting inds
    sz_, sy_, sx_ = spacing
    vec = np.stack(
        [(xx - inds[2]) * sx_, (yy - inds[1]) * sy_, (zz - inds[0]) * sz_], -1
    )
    phi = vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-6)

    geom = {
        "cell": mask,
        "tau": (2.0 * d - 1.0).astype(np.float32),
        "d": d.astype(np.float32),
        "phi": phi.astype(np.float32),
    }
    return {"geom": geom, "sl": sub, "pts": pts, "radius_vox": R, "length_vox": length}
    
def make_aggregates(tape, j, G, xyz, shape, spacing, um_per_vox, grid, L=4, l_min=2):
    r_um = 0.5 * float(G.DIAM_UM.v) * np.exp(float(G.DIAM_SIGMA.v) * tape["z_art_size"][j])
    R = max(float(r_um) / float(um_per_vox), 0.5)
    reach = int(np.ceil(R * np.exp(3.0 * float(G.ROUGH.v)))) + 2
    sl = patch_grid(xyz, reach, shape)
    if any(s.stop - s.start <= 0 for s in sl):
        return None

    centre = vox_to_world(xyz, shape, spacing)
    rho, u = body_frame(grid[sl], rot=None, centre=centre, elong=1.0)
    r_fn = make_boundary(coeff=tape["art_sh"][j], R=R, kappa=float(G.ROUGH.v), beta=2.0,
                        L=L, l_min=l_min)
    d = rho / np.maximum(r_fn(u), 1e-6)
    mask = d <= 1.0
    d = np.minimum(d, cfg._TAU_CLIP)          # see the note in make_fussel
    
    return {"geom":{
                "cell": mask, 
                "tau": (2.0 * d - 1.0).astype(np.float32),
                "d": d.astype(np.float32), 
                "phi": u.astype(np.float32)
            }, 
            "sl": sl, 
            "radius_vox": R}

def to_artifact_marker(marker, mu, width, sharp):
    """The panel marker's TEXTURE, relocalised onto the artifact's own tau."""
    comps = [dataclasses.replace(c, mu=mu, width=width, sharp=sharp)
             for c in marker.noise_components]
    return dataclasses.replace(marker, noise_components=comps)

def to_flat_artifact_marker(marker, mu, width, sharp):
    """Flat marker channel for the Fussel."""
    comp = FlatNoise(w=1.0, mu=mu, width=width, sharp=sharp)
    return dataclasses.replace(marker, noise_components=[comp])


def build_artifacts(vols, tape, cfg, AR, opt, shape, panel, um_per_vox, spacing, geom = None, tissue_support= None, 
                    pool_bank=None ,L=4, l_min=2, n_t=512
                #     , TG, shape, base_geom, spacing, Panel, CellTypes, Fractions,
                #  um_per_vox, L=4, l_min=2, rule=None, pool_bank=None
                 ):
    """Tissue in three phases: geometry, then type assignment, then appearance.
    n_t is the fussels centreline's own sampling resolution (points along the curve)"""
    names = list(panel.Markers)
    
    # get the locations for fussels / Aggregats
    F, G = AR.Fussel, AR.Aggregate
    n_fib = max(int(round(F.N.v)), 0)
    n_agg = max(int(round(G.N.v)), 0)
    
    _, fib_rows = artifact_sites(tape, AR.Map, shape, n_fib, tissue_support)
    agg_xy, agg_rows = artifact_sites(tape, AR.Map, shape, n_agg, tissue_support,
                                      skip=len(fib_rows))
    
    nid = cfg.ARTIFACT_ID0
    table, affinities = [], {}
    grid = centred_grid_3d(shape, spacing)
    base_geom = CellGeometry() if geom is None else geom
    art_labels = np.zeros(shape, np.int32)
    
    
    aff = marker_affinity(panel, names)
    
    # FUSSEL generation
    for j in fib_rows:
        fb = make_fussel(tape=tape, F=F, j=int(j), shape = shape, opt = opt, cfg = cfg, um_per_vox=um_per_vox, n_t = n_t, spacing = spacing)
        if not fb["geom"]["cell"].any():
            continue
        
        # Marker expression
        affinities[nid] = aff
        gain = float(F.GAIN.v) * np.exp(float(F.GAIN_SIGMA.v) * tape["z_art_gain"][int(j)])
        sl, gm = fb["sl"], fb["geom"]
        ptape = TapeDict(
            texture_noise=tape["texture_noise"][(slice(None),) + sl],
            gate_noise=tape["gate_noise"][(slice(None),) + sl],
        )
        off = 0
        for name in names:
            m = panel.Markers[name]
            # Compute expression based on affinity
            lvl = aff[name] * gain
            if lvl > 1e-3:
                img = render_marker(tape=ptape, marker=to_flat_artifact_marker(m, F.MU, F.WIDTH,
                                                                        F.SHARP),
                                    cell=gm, spacing=spacing, geom=base_geom,
                                    um_per_vox=um_per_vox, edge_softness=0.0,
                                    pool_offset=off, pool_bank=pool_bank)
                vols[name][sl] += np.float32(lvl) * img
            off += len(m.noise_components)
        art_labels[sl] = np.where(gm["cell"], nid, art_labels[sl])
        table.append(dict(id=nid, kind="fussel",
                        diam_um=2.0 * fb["radius_vox"] * um_per_vox, 
                        n_markers=int(sum(v > 1e-3 for v in aff.values())),
                        gain=gain))
        nid += 1
        
    ab = [n for n in names if n != cfg.DAPI_NAME]
    for k, j in enumerate(agg_rows):
        if not ab:
            # stop if there are no markers expressed
            break
        xyz = np.array([agg_xy[k][0], agg_xy[k][1], tape["u_art_z"][int(j)] * (shape[0] - 1)])
        ag = make_aggregates(tape, int(j), G, xyz, shape, spacing, um_per_vox, grid,
                            L=L, l_min=l_min)
        
        if not ag["geom"]["cell"].any(): 
            continue
        
        # Marker expression
        ch = ab[min(int(tape["u_art_ch"][int(j)] * len(ab)), len(ab) - 1)]
        gain = float(G.GAIN.v) * np.exp(float(G.GAIN_SIGMA.v) * tape["z_art_gain"][int(j)])
        sl, gm = ag["sl"], ag["geom"]
        ptape = TapeDict(texture_noise=tape["texture_noise"][(slice(None),) + sl],
                         gate_noise=tape["gate_noise"][(slice(None),) + sl])
        off = 0
        for name in names:
            m = panel.Markers[name]
            # one antibody precipitated, so it shows in ITS channel. SPILL relaxes that.
            lvl = gain if name == ch else (gain * float(G.SPILL.v) if name in ab else 0.0)
            if lvl > 1e-3:
                img = render_marker(tape=ptape, marker=to_artifact_marker(m, G.MU, G.WIDTH,
                                                                       G.SHARP),
                                    cell=gm, spacing=spacing, geom=base_geom,
                                    um_per_vox=um_per_vox, edge_softness=0.0,
                                    pool_offset=off, pool_bank=pool_bank)
                vols[name][sl] += np.float32(lvl) * img
            off += len(m.noise_components)
        art_labels[sl] = np.where(gm["cell"], nid, art_labels[sl])
        table.append(dict(id=nid, kind="aggregate", channel=ch,
                        diam_um=2.0 * ag["radius_vox"] * um_per_vox, gain=gain))
        nid += 1
        
    short = (max(n_fib - len(fib_rows), 0), max(n_agg - len(agg_rows), 0))
    return {
        "labels": art_labels,
        "table": table,
        "affinity": affinities,
        "n_fussel": sum(r["kind"] == "fussel" for r in table),
        "n_aggregate": sum(r["kind"] == "aggregate" for r in table),
        "shortfall": {"fussel": short[0], "aggregate": short[1]},
    }
    