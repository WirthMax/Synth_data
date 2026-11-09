"""flexible-mode sampling: the per-sample panel, cell-type set and compositions.

In fixed mode a dataset's semantics are frozen once (`gen_dataset.panel_structure`): the markers,
which types are positive for which marker, and which structures exist are all one draw shared by
every frame. Flexible mode (plan Phase 3) instead draws the *whole biological layout* afresh for
every sample, so that one training set spans an open-ended space of panels and cell-type
vocabularies rather than a single fixed one. This module holds the draws that do that, as pure
functions of a numpy Generator.

These are NOT render-path randomness: they play the same role as the existing per-dataset
`panel_structure` / per-sample `generate_world` draws, which also sit outside the tape. The tape
invariant (all render randomness frozen before any parameter is read) is untouched -- the frozen
per-cell draws that *place* and *type* cells (`tape["u_type"]`, `z_size`, ...) still live on the
tape; what this module produces is the theta that those frozen fields are thresholded against.

Everything here is deterministic given the Generator, so a dataset is reproducible from its seed
and the lineage / composition unit tests (plan 3.8.2, 3.8.3) can exercise the priors directly.

Units: RADIUS is in voxels (as in CellGeometry); expression `level` is stored as log10 of an
absolute brightness, the quantity that replaces the panel `amp x expression-level` product in
flexible mode. Angles and compartments are categorical.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from parameter import (CellGeometry, CellType, MarkerPanel, Ordered, P, PanelMarker,
                       TissueArchitecture, Vasculature, dapi_marker, make_nuclear, narrow,
                       NOISE_KINDS)

# --- compartment prior (plan 3.1) -------------------------------------------------------------
# where a marker localises. A per-MARKER property: every type that expresses marker m shows it in
# the same compartment, so this is drawn once per channel, not per (type, marker). The caller
# passes the subset the generator can actually render and the prior is renormalised over it, so
# dropping a compartment never changes the relative odds of the ones that remain.
COMPARTMENT_PRIOR: dict[str, float] = {
    "nuclear": 0.25,
    "membrane": 0.40,
    "cytoplasmic": 0.30,
    "punctate": 0.05,
}

# the categorical index space for the compartment (theta v3 marker.compartment is an int over
# this fixed order). It stays full even when a compartment is not currently renderable, so the
# layout the model sees never shifts when the renderable set changes.
COMPARTMENT_NAMES: tuple[str, ...] = ("nuclear", "membrane", "cytoplasmic", "punctate")

# which compartments the tau-localisation can actually render, and the (mu, width, sharp) tau band
# each maps to. tau is -1 at the nucleus centre, 0 at the nuclear envelope, +1 at the plasma
# membrane (invariant 5). nuclear sits below the envelope (and is forced there by make_nuclear);
# membrane is a sharp band at the rim; cytoplasmic fills the space between. `punctate` is OMITTED
# for now -- it needs a dedicated spot texture rather than a tau band, so per plan 3.1 it is left
# out of the renderable set and the prior is renormalised over the rest. # VERIFY once a punctate
# texture exists, then add it here.
COMPARTMENT_BAND: dict[str, dict] = {
    "nuclear":     dict(mu=-0.50, width=0.60, sharp=4.0, nuclear=True),
    "membrane":    dict(mu=1.00,  width=0.30, sharp=6.0, nuclear=False),
    "cytoplasmic": dict(mu=0.40,  width=0.70, sharp=3.0, nuclear=False),
}
RENDERABLE_COMPARTMENTS: tuple[str, ...] = ("nuclear", "membrane", "cytoplasmic")

# A marker's localisation is NOT a 3-way category -- it is a continuous position of its intensity
# band along tau (-1 nucleus centre, 0 envelope, +1 plasma membrane). That position is `mu`, and it
# is what the model regresses. We sample it continuously over [MU_LO, MU_HI] (nucleus interior ->
# membrane); the old compartment classes are just thresholds of this (and survive only as a
# diagnostic readout). The band's shape (width/sharp) is tied to mu by a smooth map -- broad & soft
# in the nucleus, tight & sharp at the membrane -- so `mu` is the SINGLE localisation knob and there
# is no separate unpredicted nuisance.
import config as cfg
MU_LO, MU_HI, DAPI_MU = cfg.MU_LO, cfg.MU_HI, cfg.DAPI_MU   # single source of truth (config.py)


def band_from_mu(mu):
    """(mu, width, sharp) for a localisation band centred at `mu`. width/sharp interpolate from
    broad/soft near the nucleus to tight/sharp at the membrane."""
    t = float(np.clip((mu - MU_LO) / (MU_HI - MU_LO), 0.0, 1.0))
    width = 0.65 + t * (0.30 - 0.65)
    sharp = 3.0 + t * (6.0 - 3.0)
    return float(mu), width, sharp


def compartment_of_mu(mu):
    """Nearest discrete compartment for a given mu -- diagnostic only (not used in generation)."""
    return min(COMPARTMENT_BAND, key=lambda c: abs(COMPARTMENT_BAND[c]["mu"] - float(mu)))

# the five geometry scalars that make up a type's shape in theta v3 (plan 3.6), in that order.
GEOM_KEYS: tuple[str, ...] = ("RADIUS", "ELONG", "ROUGH", "BETA", "NUC_FRAC")

# SD of the per-child geometry mutation in logit space (plan 3.2: "small noise in logit space").
# The plan gives log-RADIUS its own N(0, 0.15); the other four share this. Chosen, not measured:
# 0.3 logit units moves a mid-range scalar by roughly +-7% of its full span. # VERIFY against the
# visual sheet once Phase 3.8 renders it.
GEOM_LOGIT_SD: float = 0.30
RADIUS_LOG_SD: float = 0.15          # plan 3.2, applied to log RADIUS directly


def _bounds(holder, name):
    """(lo, hi) of a declared P on a ParamHolder class, read off the class so the prior can never
    drift from the fittable range it will later be normalised against (invariant 2)."""
    p = getattr(holder, name)
    return float(p.lo), float(p.hi)


def expression_range():
    """[b_min, b_max] absolute brightness, derived from the panel `amp` prior (plan 3.2):
    b_min = 0.3 x smallest amp, b_max = 3 x largest amp. In flexible mode expression is stored as
    this absolute brightness (its log10), which REPLACES the per-marker `amp` x level product --
    so the two together can no longer trade off against each other and leave the image unchanged.
    """
    lo, hi = _bounds(PanelMarker, "amp")
    return 0.3 * lo * cfg.EXPR_GAIN, 3.0 * hi * cfg.EXPR_GAIN


def _geometry_bounds():
    """Sampling bounds for the five geometry scalars. RADIUS/ELONG are the ranges confirmed for
    this FOV (RADIUS log-uniform over its full voxel range; ELONG [1, 7], narrower than the
    class's [0.3, 15] prolate/oblate span because a tile this size never shows extreme oblates);
    the rest keep their class bounds."""
    g = CellGeometry()
    lo = {k: _bounds(g, k)[0] for k in GEOM_KEYS}
    hi = {k: _bounds(g, k)[1] for k in GEOM_KEYS}
    lo["ELONG"], hi["ELONG"] = 1.0, 7.0          # confirmed with the user; subset of [0.3, 15]
    return lo, hi


def _sample_geometry(rng):
    """One type's geometry drawn from the prior. RADIUS is log-uniform (a radius is a scale, so
    equal odds per decade, not per voxel); the others are uniform over their bounds."""
    lo, hi = _geometry_bounds()
    g = {}
    g["RADIUS"] = float(np.exp(rng.uniform(np.log(lo["RADIUS"]), np.log(hi["RADIUS"]))))
    for k in ("ELONG", "ROUGH", "BETA", "NUC_FRAC"):
        g[k] = float(rng.uniform(lo[k], hi[k]))
    return g


def _logit(u):
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    return np.log(u / (1.0 - u))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _mutate_geometry(rng, g):
    """A child's geometry: the parent's, nudged. log RADIUS gets its own Gaussian; the other four
    are perturbed in logit space (so a scalar near a bound moves less and never leaves it) and
    clamped back to their bounds."""
    lo, hi = _geometry_bounds()
    out = dict(g)
    out["RADIUS"] = float(np.clip(g["RADIUS"] * np.exp(RADIUS_LOG_SD * rng.standard_normal()),
                                  lo["RADIUS"], hi["RADIUS"]))
    for k in ("ELONG", "ROUGH", "BETA", "NUC_FRAC"):
        span = hi[k] - lo[k]
        u = (g[k] - lo[k]) / (span + 1e-12)
        u = _sigmoid(_logit(u) + GEOM_LOGIT_SD * rng.standard_normal())
        out[k] = float(lo[k] + u * span)
    return out


@dataclass
class TypeSet:
    """A sampled cell-type vocabulary over a panel of M markers.

    `positive[k, m]` is whether type k expresses marker m; `level[k, m]` is log10 of its absolute
    brightness where positive and NaN where not (undefined, never read). `geometry[k]` is the five
    GEOM_KEYS scalars. `parent[k]` is the lineage parent (-1 for a root), kept so a test or a
    diagnostic can see the tree. K is the realised number of types (may be below the requested
    K_true when the tree cannot grow further without violating distinctness or max_depth).
    """
    positive: np.ndarray            # bool [K, M]
    level: np.ndarray               # float [K, M], log10 brightness, NaN where not positive
    geometry: np.ndarray            # float [K, 5], columns = GEOM_KEYS
    parent: np.ndarray              # int [K]
    counterstain: int               # marker index of the always-present nuclear counterstain

    @property
    def K(self):
        return self.positive.shape[0]

    @property
    def M(self):
        return self.positive.shape[1]


def _geom_vec(g):
    return np.array([g[k] for k in GEOM_KEYS], float)


def are_distinct(a: int, b: int, positive, level, geometry, min_level_diff):
    """The plan 3.2 distinctness rule. Two types are distinct if ANY of:
      - they differ in the positivity of >= 1 marker,
      - a marker both express differs in level by >= `min_level_diff` decades,
      - their log RADIUS differs by >= 0.2,
      - their ELONG differs by >= 0.5.
    `min_level_diff` is the curriculum's shrinking margin (plan 3.7: 0.6 -> 0.3 decades). Levels
    are already stored as log10, so a decade is a unit difference.
    """
    if np.any(positive[a] != positive[b]):
        return True
    shared = positive[a] & positive[b]
    if np.any(shared) and np.nanmax(np.abs(level[a][shared] - level[b][shared])) >= min_level_diff:
        return True
    ra, rb = geometry[a, 0], geometry[b, 0]
    if abs(np.log(ra) - np.log(rb)) >= 0.2:
        return True
    if abs(geometry[a, 1] - geometry[b, 1]) >= 0.5:
        return True
    return False


def sample_typeset(rng, M, *, k_true_range=(1, 30), max_depth=4, min_level_diff=0.3,
                   p_pos=0.25, counterstain=0, max_retries=20, background=None):
    """Draw a cell-type set from the lineage-tree prior (plan 3.2).

    Root types are independent: each marker (bar the counterstain) is positive with probability
    `p_pos` and, if positive, takes a log10-uniform level over the expression range. The
    counterstain is positive in every type (every cell has a nucleus). New types are then grown by
    copying an existing parent and mutating it -- flipping a few markers, rescaling a few levels,
    nudging the geometry -- which deliberately yields both near-twins and very different types in
    one tile so the model has to learn fine distinctions. A candidate that is not distinct from
    every existing type (`are_distinct`) is rejected and redrawn, up to `max_retries`; if the tree
    cannot grow a new distinct type it stops early, so the realised K may be below K_true.

    `max_depth` caps the tree depth (root = depth 1), the curriculum's complexity knob: at
    max_depth 1 only roots exist and K = R.
    """
    b_min, b_max = expression_range()
    lg_lo, lg_hi = np.log10(b_min), np.log10(b_max)
    bg = np.zeros(M, bool) if background is None else np.asarray(background, bool)

    k_true = int(rng.integers(k_true_range[0], k_true_range[1] + 1))
    n_roots = int(rng.integers(1, min(5, k_true) + 1))

    positive = np.zeros((0, M), bool)
    level = np.zeros((0, M), float)
    geometry = np.zeros((0, len(GEOM_KEYS)), float)
    parent = np.zeros((0,), int)
    depth = np.zeros((0,), int)

    def draw_level():
        return float(rng.uniform(lg_lo, lg_hi))

    def add(pos_row, lvl_row, geom, par, dep):
        nonlocal positive, level, geometry, parent, depth
        positive = np.vstack([positive, pos_row[None]])
        level = np.vstack([level, lvl_row[None]])
        geometry = np.vstack([geometry, _geom_vec(geom)[None]])
        parent = np.append(parent, par)
        depth = np.append(depth, dep)

    def distinct_from_all(pos_row, lvl_row, geom):
        pos = np.vstack([positive, pos_row[None]])
        lvl = np.vstack([level, lvl_row[None]])
        geo = np.vstack([geometry, _geom_vec(geom)[None]])
        j = pos.shape[0] - 1
        return all(are_distinct(i, j, pos, lvl, geo, min_level_diff) for i in range(j))

    # roots
    for _ in range(n_roots):
        for _try in range(max_retries):
            pos_row = rng.random(M) < p_pos
            pos_row[counterstain] = True
            pos_row[bg] = False                              # background-only markers: never expressed
            lvl_row = np.full(M, np.nan)
            lvl_row[pos_row] = [draw_level() for _ in range(int(pos_row.sum()))]
            geom = _sample_geometry(rng)
            if distinct_from_all(pos_row, lvl_row, geom):
                add(pos_row, lvl_row, geom, -1, 1)
                break

    # children
    while positive.shape[0] < k_true:
        eligible = np.flatnonzero(depth < max_depth)
        if eligible.size == 0:
            break
        grew = False
        for _try in range(max_retries):
            par = int(rng.choice(eligible))
            pos_row = positive[par].copy()
            lvl_row = level[par].copy()
            geom = {k: geometry[par, i] for i, k in enumerate(GEOM_KEYS)}

            # flip 1 + Poisson(1) markers (never the counterstain or a background marker)
            n_flip = 1 + int(rng.poisson(1))
            cand = np.array([m for m in range(M) if m != counterstain and not bg[m]])
            flip = rng.choice(cand, size=min(n_flip, cand.size), replace=False)
            for m in flip:
                pos_row[m] = not pos_row[m]
                lvl_row[m] = draw_level() if pos_row[m] else np.nan

            # rescale Poisson(1) currently-positive markers by 10^N(0, 0.25)
            pos_idx = np.flatnonzero(pos_row)
            n_scale = min(int(rng.poisson(1)), pos_idx.size)
            if n_scale:
                for m in rng.choice(pos_idx, size=n_scale, replace=False):
                    lvl_row[m] = float(np.clip(lvl_row[m] + 0.25 * rng.standard_normal(),
                                               lg_lo, lg_hi))

            geom = _sample_geometry(rng) if rng.random() < 0.1 else _mutate_geometry(rng, geom)

            if distinct_from_all(pos_row, lvl_row, geom):
                add(pos_row, lvl_row, geom, par, depth[par] + 1)
                grew = True
                break
        if not grew:
            break

    return TypeSet(positive=positive, level=level, geometry=geometry,
                   parent=parent, counterstain=counterstain)


def sample_panel(rng, *, m_range=(8, 40), dyes, n_noise_kinds, noise_comps=(1, 4),
                 p_background=0.0):
    """Draw a panel of M markers (plan 3.1).

    M ~ U{m_range}. Channel 0 is the nuclear counterstain -- always present, nuclear, the
    registration reference -- the canonical, UNPERMUTED order. Every other channel draws a dye, a
    continuous localisation `mu` in [MU_LO, MU_HI] and a handful of noise texture kinds. `order` is a
    per-sample permutation of the output channel positions so that position carries no information.

    `background[m]` (prob `p_background` per antibody channel; never the counterstain) flags a
    BACKGROUND-ONLY marker -- a channel that no cell type expresses, so it shows only per-dye
    autofluorescence + noise (+ aggregates when artifacts are on). Real panels are full of these
    (non-binding antibodies, failed stains); the model must learn to ignore them rather than read
    cells from them.
    """
    M = int(rng.integers(m_range[0], m_range[1] + 1))
    mu = [DAPI_MU]                                          # the counterstain
    dye = [None]                                            # filled by the caller's DAPI marker
    noise = [()]                                            # DAPI carries no antibody texture
    background = [False]                                    # the counterstain is never background
    for _ in range(M - 1):
        mu.append(float(rng.uniform(MU_LO, MU_HI)))
        dye.append(str(rng.choice(list(dyes))))
        k = int(rng.integers(noise_comps[0], noise_comps[1] + 1))
        noise.append(tuple(int(x) for x in rng.integers(0, n_noise_kinds, size=k)))
        background.append(bool(rng.random() < p_background))

    order = rng.permutation(M)
    return dict(M=M, mu=mu, dye=dye, noise=noise, order=order, counterstain=0,
                background=np.array(background, bool))


def sample_compositions(rng, kinds, K, *, p_allow_bg=0.8, p_allow_struct=0.4):
    """Draw one composition per context (plan 3.4).

    Contexts are [background] + the distinct structure KINDS present (per-kind, not per-instance:
    compositions replace the old per-instance Profile log-weights, which had a free additive
    direction and so hid information). Each type is allowed in background with probability
    `p_allow_bg` and in each structure kind with `p_allow_struct`; every context is guaranteed at
    least one allowed type. `pi[c]` is Dirichlet(1) over the allowed types and exactly 0 on the
    rest, so it is a genuine probability vector with nothing hidden.

    Returns (contexts, allowed, pi): the context names, the bool[C, K] allow mask, and the
    float[C, K] composition matrix whose rows sum to 1.
    """
    contexts = ["background"] + list(dict.fromkeys(kinds))   # dedupe, keep order
    C = len(contexts)
    allowed = np.zeros((C, K), bool)
    pi = np.zeros((C, K), float)
    for c, name in enumerate(contexts):
        p = p_allow_bg if name == "background" else p_allow_struct
        row = rng.random(K) < p
        if not row.any():
            row[rng.integers(K)] = True                      # every context needs a type
        allowed[c] = row
        a = np.flatnonzero(row)
        pi[c, a] = rng.dirichlet(np.ones(a.size))
    return contexts, allowed, pi


def assign_from_compositions(w, pi, u):
    """Assign a type to each cell by the plan 3.4 rule: p(type k | x) = sum_c w_c(x) . pi_c[k],
    then inverse-CDF sample with the cell's frozen uniform `u` (the tape's `u_type`). `w` is the
    per-cell context membership (N, C), rows summing to 1 (one-hot if membership is hard); `pi` is
    (C, K); `u` is (N,). Returns the chosen type index per cell (N,).

    This is the same searchsorted-on-a-cumulative rule the fixed path uses, so a frozen `u` maps
    smoothly through the compositions -- moving a composition entry moves cells between types
    without resampling anything.
    """
    w = np.asarray(w, float)
    p = w @ pi                                               # (N, K)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    cuts = np.cumsum(p, axis=1)
    u = np.clip(np.asarray(u, float), 0.0, 1.0 - 1e-9)
    return np.array([int(np.searchsorted(cuts[i], u[i])) for i in range(len(u))])


# --- building the sampled layout into real objects --------------------------------------------
# A type name from its index; T000, T001, ... in the TypeSet's own (meaningless) order.
def _type_name(k):
    return f"T{k:03d}"


# A marker name from its CANONICAL index; m0 is always the counterstain (DAPI). The channel ORDER
# in the image is a separate per-sample permutation, so these names index biology, not position.
def _marker_name(m, dapi_name):
    return dapi_name if m == 0 else f"M{m:02d}"


def build_marker(rng, *, name, dye, mu, noise_kind_idx):
    """One antibody channel. Its texture components are drawn from the existing noise priors (they
    are theta), but its tau localisation is set by the continuous `mu` (plus width/sharp tied to it
    via `band_from_mu`) on every component, so the marker shows the same way in every type that
    expresses it. `amp` is pinned to 1.0: in flexible mode the absolute brightness lives entirely in
    each type's expression level, so the old amp x level product (which could trade off invisibly)
    is gone.
    """
    comps = [NOISE_KINDS[k]() for k in noise_kind_idx]
    mu_v, width_v, sharp_v = band_from_mu(mu)
    for c in comps:
        c.mu = narrow(c.mu, c.mu.lo, c.mu.hi, mu_v)
        c.width = narrow(c.width, c.width.lo, c.width.hi, width_v)
        c.sharp = narrow(c.sharp, c.sharp.lo, c.sharp.hi, sharp_v)
    m = PanelMarker(name=name, fluorophore=dye, noise_components=comps)
    m.amp = narrow(m.amp, 1.0, 1.0, 1.0)                     # pinned; brightness is in expression
    return m


def _flexible_architecture(rng, *, max_structures=3):
    """A small tissue architecture for flexible mode: 0..max_structures bundles/vessels of random
    kind. ZERO is now reachable -- a structure-free, cell-only (or, with COVER=0, empty) tile, which
    build_architecture handles by uniform thinning with no director. Profiles are left empty --
    compositions (per kind) replace the per-instance log-weights, so no structure needs a Profile any
    more. FRAC/WALL_IN are narrowed and CONTACT pinned exactly as the fixed-mode builder does, for the
    same feasibility reasons (a lumen must not outgrow the tissue budget; the compliance solve must
    not engage)."""
    n = int(rng.integers(0, max_structures + 1))
    structures, kinds = [], []
    if n > 0:
        frac_hi = 0.9 / n
        for i in range(n):
            kind = "vessel" if rng.random() < 0.5 else "ordered"
            cls = Vasculature if kind == "vessel" else Ordered
            st = cls(name=f"s{i}", Profile={})
            st.FRAC = narrow(st.FRAC, st.FRAC.lo, frac_hi * (0.5 if kind == "vessel" else 1.0))
            if kind == "vessel":
                st.WALL_IN = narrow(st.WALL_IN, st.WALL_IN.lo, min(st.WALL_IN.hi, 0.5))
            structures.append(st)
            kinds.append(kind)
    arch = TissueArchitecture(Structures=structures)
    arch.CONTACT = narrow(arch.CONTACT, 0.0, 0.0)
    return arch, kinds


def _expected_visible_cells(tg):
    """Rough estimate of how many cells a tile of this density will actually SHOW. Used only to cap
    the requested type count so a sparse tile does not ask for more distinct types than it can
    realise (each present type needs several visible cells). Anchored at the historical prior
    (MIN_DIST=11, COVER=0.75 -> ~40 visible cells) and scaled inverse-square in spacing and linearly
    in coverage. Deliberately approximate -- it bounds a count, it is not a target."""
    md = max(float(tg.MIN_DIST.v), 1e-3)
    return 40.0 * (float(tg.COVER.v) / 0.75) * (11.0 / md) ** 2


def build_flexible_world(rng, *, curriculum=None, dyes, n_noise_kinds=len(NOISE_KINDS),
                         dapi_name="DAPI", max_structures=3, detector=None):
    """Draw a whole flexible-mode sample: the panel, the lineage cell-type set, the tissue
    architecture and the per-context compositions, assembled into the same world dict that
    `render_frame` already consumes (PANEL / TYPES / ARCH / CELL / TG / FRACTIONS), plus the extra
    flexible-mode fields COMPOSITIONS, PANEL_SPEC, TYPESET and META that carry the sample's layout
    for theta v3 and the data loader.

    `curriculum` is the plan 3.7 knob dict (m_range, k_true_range, max_depth, min_level_diff);
    None uses the final prior. The order in which markers appear in the panel (and so in the image)
    is a per-sample permutation, so channel position carries no information.
    """
    from parameter import TissueGeometry, narrow as _narrow  # local: avoids a heavy import at load
    import config as _cfg
    cur = dict(m_range=(8, 40), k_true_range=(1, 30), max_depth=4, min_level_diff=0.3,
               p_background=0.0)
    cur.update(curriculum or {})

    # --- per-tile tissue density, drawn BEFORE the type set so the type count can be capped to what
    # the cellularity can actually show. The model must handle the whole span (dense-no-background ->
    # empty) at every round, so density is NOT interpolated by the curriculum. An empty tile is
    # COVER=0 (support_mask returns all-False) with zero structures -- a genuine zero-cell negative. ---
    dens = _cfg.DENSITY
    empty = bool(rng.random() < float(dens.get("p_empty", 0.0)))
    ms = 0 if empty else int(dens["max_structures"])
    arch, kinds = _flexible_architecture(rng, max_structures=ms)

    tg = TissueGeometry()
    if empty:
        tg.COVER = _narrow(tg.COVER, tg.COVER.lo, tg.COVER.hi, 0.0)
        tg.MIN_DIST = _narrow(tg.MIN_DIST, tg.MIN_DIST.lo, tg.MIN_DIST.hi, dens["MIN_DIST"][1])
    else:
        # a vessel lumen is non-tissue, so a near-full COVER plus a vessel is geometrically infeasible
        # (tissue + lumen > volume, which build_architecture rejects). Cap COVER when a vessel is
        # present; a structure-free or ordered-only tile can still fill the frame (dense, no-background).
        cov_lo, cov_hi = dens["COVER"]
        if "vessel" in kinds:
            cov_hi = min(cov_hi, 0.8)
        tg.MIN_DIST = _narrow(tg.MIN_DIST, tg.MIN_DIST.lo, tg.MIN_DIST.hi,
                              float(rng.uniform(*dens["MIN_DIST"])))
        tg.COVER = _narrow(tg.COVER, tg.COVER.lo, tg.COVER.hi, float(rng.uniform(cov_lo, cov_hi)))
        tg.GROW = _narrow(tg.GROW, tg.GROW.lo, tg.GROW.hi, float(rng.uniform(*dens["GROW"])))

    # cap the requested number of distinct types to what this cellularity can realise: a type counts
    # as present only with >= VIS.min_visible_cells visible cells, so a sparse tile that asked for many
    # types would end up with ZERO present types (no typing signal) and bias the existence head toward
    # "no types". Dense tiles are unaffected (the cap exceeds the curriculum ceiling there).
    klo, khi = cur["k_true_range"]
    k_cap = max(1, int(round(_expected_visible_cells(tg) / float(_cfg.VIS["min_visible_cells"]))))
    k_range = (min(klo, k_cap), min(khi, k_cap))

    panel_spec = sample_panel(rng, m_range=cur["m_range"], dyes=dyes, n_noise_kinds=n_noise_kinds,
                              p_background=cur["p_background"])
    M = panel_spec["M"]
    bg = panel_spec["background"]
    ts = sample_typeset(rng, M, k_true_range=k_range, max_depth=cur["max_depth"],
                        min_level_diff=cur["min_level_diff"], counterstain=panel_spec["counterstain"],
                        background=bg)
    K = ts.K
    b_min, b_max = expression_range()

    # markers in CANONICAL order, then inserted into the panel in the permuted channel order
    canon = {}
    for m in range(M):
        nm = _marker_name(m, dapi_name)
        if m == panel_spec["counterstain"]:
            canon[nm] = dapi_marker(name=nm)                 # the counterstain, always nuclear
        else:
            mk = build_marker(rng, name=nm, dye=panel_spec["dye"][m], mu=panel_spec["mu"][m],
                              noise_kind_idx=panel_spec["noise"][m])
            if bg[m]:                                        # background-only -> antibody aggregates
                mk.artifact_affinity = _narrow(mk.artifact_affinity, 1.0,
                                               mk.artifact_affinity.hi, mk.artifact_affinity.hi)
            canon[nm] = mk
    names_canon = [_marker_name(m, dapi_name) for m in range(M)]
    order = panel_spec["order"]
    panel = MarkerPanel(Markers={names_canon[p]: canon[names_canon[p]] for p in order})

    # cell types: geometry from the TypeSet, expression as ABSOLUTE brightness (10^level) on the
    # positive markers. amp is pinned to 1, so render_marker's amp x level is exactly the brightness.
    types = {}
    for k in range(K):
        geom = CellGeometry(**{key: float(ts.geometry[k, i]) for i, key in enumerate(GEOM_KEYS)})
        expr = {}
        for m in np.flatnonzero(ts.positive[k]):
            nm = names_canon[m]
            bright = float(10.0 ** ts.level[k, m])
            expr[nm] = P(0.0, b_max, 0.05, bright, f"expr {nm}",
                         comment="absolute brightness of this marker in this type (flexible mode); "
                                 "replaces the panel amp x level product.")
        types[_type_name(k)] = CellType(name=_type_name(k), Color="black",
                                        Geometry=geom, Expression=expr)

    contexts, allowed, pi = sample_compositions(rng, kinds, K)
    struct_to_ctx = [contexts.index(kd) for kd in kinds]
    compositions = dict(contexts=contexts, pi=pi, allowed=allowed,
                        struct_to_ctx=struct_to_ctx, type_order=[_type_name(k) for k in range(K)])

    world = {
        "PANEL": panel, "TYPES": types, "TG": tg, "DET": detector,
        "ARCH": arch, "CELL": CellGeometry(), "FRACTIONS": list(np.ones(K) / K),
        "COMPOSITIONS": compositions, "PANEL_SPEC": panel_spec, "TYPESET": ts,
        "META": dict(K=K, M=M, counterstain=panel_spec["counterstain"],
                     channel_order=order, b_range=(b_min, b_max),
                     mu=list(panel_spec["mu"]),                 # canonical-order localisation targets
                     empty=empty, n_struct=len(arch.Structures)),
    }
    return world


# --- theta v3 (plan 3.6): the sample's fittable layout as a flat dict of arrays ----------------
# What the inversion model is trained to recover. A FLAT dict of numpy arrays, so it drops
# straight into np.savez and loads back bit-for-bit. `theta_v3_nested` regroups it into the
# global/marker/types/contexts/meta view the plan writes; padding each sample to a common
# (K_max, M_max) tensor for the model is deliberately left to Phase 4. Every bounded continuous
# scalar is stored NORMALISED to [0, 1] by its prior bounds (clamped to [1e-4, 1-1e-4]); positive /
# compartment / present are categorical; pi is a simplex (rows sum to 1), stored as-is.
_THETA_CLAMP = (1e-4, 1.0 - 1e-4)
# ARCH scalars that are not fitted even in flexible mode (compliance-solve stability), matching
# gen_dataset.ARCH_PIN; Profile is gone entirely so needs no pin.
_GLOBAL_PIN = (r"\.STIFF$", r"\.ANISO$", r"^CONTACT$")
_NOISE_KEEP = (r"\.scale$", r"\.clust$", r"\.fill$", r"\.s$")


def _clamp_u(u):
    return np.clip(u, *_THETA_CLAMP)


_NOISE4_KEYS = ("scale", "clust", "fill", "s")


def _marker_noise4(panel):
    """Per-marker [M, 4] noise target (scale, clust, fill, s), in IMAGE-channel order, normalised to
    [0, 1]. The theta v3 marker.noise tensor the per-marker head predicts: one descriptor per
    channel, read off that channel's first noise component (0.5 -- the neutral mid -- for a subfield
    the component's noise kind does not carry). A lossy summary of a possibly multi-component texture,
    but it matches the plan's declared [M, P_noise] shape."""
    from parameter import P, to_u
    markers = list(panel.Markers.values())                   # dict insertion order == image order
    out = np.full((len(markers), len(_NOISE4_KEYS)), 0.5, np.float64)
    for c, mk in enumerate(markers):
        comp = mk.noise_components[0] if mk.noise_components else None
        if comp is None:
            continue
        for j, key in enumerate(_NOISE4_KEYS):
            p = getattr(comp, key, None)
            if isinstance(p, P):
                out[c, j] = _clamp_u(to_u(p))
    return out


def _scalar_block(root, prefix, pin=(), keep=None):
    """Normalised (names, u) for every fittable P under `root`, path-prefixed by `prefix`. Reuses
    the same fitted_paths / to_u machinery the fixed theta vector uses, so a scalar normalises
    identically in both modes."""
    from parameter import fitted_paths, get_p, to_u
    paths = fitted_paths(root, pin=pin, keep=keep)
    names = np.array([f"{prefix}.{p}" for p in paths])
    u = np.array([_clamp_u(to_u(get_p(root, p))) for p in paths], np.float64)
    return names, u


def _geometry_u(geometry):
    """Normalise the [K, 5] geometry block column-by-column, matching the sampling bounds (RADIUS
    log, the rest linear; ELONG over [1, 7])."""
    lo, hi = _geometry_bounds()
    out = np.zeros_like(geometry, np.float64)
    out[:, 0] = (np.log(geometry[:, 0]) - np.log(lo["RADIUS"])) / (
        np.log(hi["RADIUS"]) - np.log(lo["RADIUS"]))
    for i, k in enumerate(("ELONG", "ROUGH", "BETA", "NUC_FRAC"), start=1):
        out[:, i] = (geometry[:, i] - lo[k]) / (hi[k] - lo[k] + 1e-12)
    return _clamp_u(out)


def theta_v3(world, cyc=None):
    """Extract the theta v3 flat dict from a flexible-mode world (plan 3.6).

    Type and marker arrays are in IMAGE-channel / type order (the sample's own internal order,
    which carries no meaning -- the model is trained invariant to it); `channel_order[c]` records
    which canonical marker sits in image channel c, so the image and theta stay relatable. `cyc`,
    if given, is render_frame's cycle side-channel; its per-channel registration offsets go into
    meta (zeros otherwise).
    """
    ts, comp, meta = world["TYPESET"], world["COMPOSITIONS"], world["META"]
    K, M = meta["K"], meta["M"]
    order = np.asarray(meta["channel_order"], int)             # image channel c -> canonical marker
    b_min, b_max = meta["b_range"]
    lg_lo, lg_hi = np.log10(b_min), np.log10(b_max)

    # types, reordered canonical-marker -> image-channel
    positive = ts.positive[:, order].astype(bool)
    level_u = np.zeros((K, M), np.float64)
    lv = (ts.level[:, order] - lg_lo) / (lg_hi - lg_lo)
    level_u[positive] = _clamp_u(lv[positive])                 # defined only where positive
    geometry_u = _geometry_u(ts.geometry)
    mu = np.asarray(meta["mu"], float)[order]                  # image order
    mu_u = _clamp_u((mu - MU_LO) / (MU_HI - MU_LO))            # normalised localisation target [M]

    g_names, g_u = _scalar_block(world["ARCH"], "ARCH", pin=_GLOBAL_PIN)
    tg_names, tg_u = _scalar_block(world["TG"], "TG", keep=(r"\.SIZE_SIGMA$",))  # per-cell size CV
    n_names, n_u = _scalar_block(world["PANEL"], "PANEL", keep=_NOISE_KEEP)

    # DENSITY block, PREPENDED so it sits at fixed global indices 0..3. The global target is padded
    # positionally (targets_v3.build_sample_targets), and the ARCH block length varies with the
    # structure count, so anything after it lands at a shifting index and is hard to learn. Density
    # is the user-facing prediction, so it goes first: cell spacing / coverage / confluence, plus the
    # total structure territory fraction (0 when there are no structures).
    from parameter import get_p, to_u
    tgw = world["TG"]
    struct_frac = float(np.clip(sum(float(s.FRAC.v) for s in world["ARCH"].Structures), 0.0, 1.0))
    dens_names = np.array(["TG.MIN_DIST", "TG.COVER", "TG.GROW", "ARCH.STRUCT_FRAC"])
    dens_u = np.array([_clamp_u(to_u(get_p(tgw, "MIN_DIST"))),
                       _clamp_u(to_u(get_p(tgw, "COVER"))),
                       _clamp_u(to_u(get_p(tgw, "GROW"))),
                       _clamp_u(struct_frac)], np.float64)
    noise4 = _marker_noise4(world["PANEL"])                   # [M, 4] target the marker head predicts

    offsets = np.zeros((M, 2), np.float32)
    if cyc is not None and "channel_offsets" in cyc:
        offsets = np.asarray(cyc["channel_offsets"], np.float32).reshape(M, 2)

    return {
        "global_names": np.concatenate([dens_names, g_names, tg_names]),
        "global_u": np.concatenate([dens_u, g_u, tg_u]),
        "noise_names": n_names,
        "noise_u": n_u,
        "noise4": noise4,
        "mu_u": mu_u,
        "positive": positive,
        "level_u": level_u,
        "geometry_u": geometry_u,
        "pi": np.asarray(comp["pi"], np.float64),
        "allowed": np.asarray(comp["allowed"], bool),
        "contexts": np.array(comp["contexts"]),
        "channel_order": order.astype(np.int64),
        "present": np.ones(K, bool),                           # all real pre-render; padding = P4
        "channel_offsets": offsets,
        "K": np.int64(K),
        "M": np.int64(M),
    }


def theta_v3_nested(flat):
    """Regroup the flat theta v3 dict into the plan 3.6 global/marker/types/contexts/meta view."""
    return {
        "global": dict(zip(flat["global_names"].tolist(), flat["global_u"].tolist())),
        "marker": {"noise_names": flat["noise_names"], "noise_u": flat["noise_u"],
                   "mu_u": flat["mu_u"]},
        "types": {"positive": flat["positive"], "level": flat["level_u"],
                  "geometry": flat["geometry_u"]},
        "contexts": {"pi": flat["pi"], "allowed": flat["allowed"], "names": flat["contexts"]},
        "meta": {"K": int(flat["K"]), "M": int(flat["M"]), "present": flat["present"],
                 "channel_order": flat["channel_order"], "channel_offsets": flat["channel_offsets"]},
    }
