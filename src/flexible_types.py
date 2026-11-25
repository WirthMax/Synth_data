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


# ELONG sampling range in force. None = the wide prior's (cfg.WIDE_PRIOR); build_flexible_world
# sets it to the run12 range for the duration of a prior="run12" draw. Theta normalisation always
# runs outside a draw, so it always uses the wide range -- one normalisation for every new dataset.
_ELONG_OVERRIDE = None
# Wide prior: the tile's cell-size scale (vox) while one world is being drawn, or None. Root types
# draw their RADIUS around it instead of log-uniform over the whole range -- see
# config.WIDE_PRIOR["CELL_RADIUS"]. Set and cleared by build_flexible_world.
_RADIUS_CENTRE = None

# geometry scalars that are SCALES: sampled, mutated and normalised in log. ELONG joined RADIUS on
# 2026-11-16, when it became the true aspect ratio and its prior log-uniform -- normalised linearly
# over [0.5, 10], three quarters of all types would sit in the bottom third of the target range.
_GEOM_LOG = ("RADIUS", "ELONG")


def _logu(rng, lo, hi):
    """Log-uniform draw: equal odds per octave, the prior for anything that is a scale."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _geometry_bounds():
    """Sampling AND normalisation bounds of the five geometry scalars. RADIUS keeps its class
    bounds (voxels); ELONG is cfg.WIDE_PRIOR["ELONG"] = [0.5, 10], the true long:short ratio (it
    was [0.6, 12] of the old elong^1.5 stretch, i.e. needles up to 41:1); the rest keep their class
    bounds."""
    g = CellGeometry()
    lo = {k: _bounds(g, k)[0] for k in GEOM_KEYS}
    hi = {k: _bounds(g, k)[1] for k in GEOM_KEYS}
    lo["ELONG"], hi["ELONG"] = _ELONG_OVERRIDE or cfg.WIDE_PRIOR["ELONG"]
    return lo, hi


def _geom_u(k, v, lo, hi):
    """One geometry scalar -> [0, 1] over its bounds, in log for the scale-like ones."""
    if k in _GEOM_LOG:
        return (np.log(np.maximum(v, 1e-12)) - np.log(lo)) / (np.log(hi) - np.log(lo) + 1e-12)
    return (v - lo) / (hi - lo + 1e-12)


def _geom_from_u(k, u, lo, hi):
    if k in _GEOM_LOG:
        return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
    return lo + u * (hi - lo)


def geometry_from_u(u):
    """Normalised geometry [..., 5] -> physical values, the inverse of `_geometry_u`: RADIUS in
    voxels, ELONG as the true long:short ratio. The one place a prediction is denormalised."""
    lo, hi = _geometry_bounds()
    u = np.asarray(u, np.float64)
    return np.stack([_geom_from_u(k, u[..., i], lo[k], hi[k]) for i, k in enumerate(GEOM_KEYS)], -1)


def _sample_geometry(rng, free_radius=False):
    """One type's geometry drawn from the prior.

    run12 prior: RADIUS log-uniform, the other four uniform over their bounds.
    Wide prior (a tile scale is set): RADIUS around the tile scale -- or, for a root flagged
    `free_radius`, anywhere in the prior with probability p_free_radius, which is what puts a
    small interstitial type beside a giant one; ELONG a mixture (5% oblate, p_round near-round,
    the rest log-uniform up to the bound); and the NUCLEUS sized in microns, NUC_FRAC following
    from it, so a big cell has a small nucleus fraction and a small cell is nearly all nucleus."""
    lo, hi = _geometry_bounds()
    g = {}
    if _RADIUS_CENTRE is None:
        g["RADIUS"] = float(np.exp(rng.uniform(np.log(lo["RADIUS"]), np.log(hi["RADIUS"]))))
        for k in ("ELONG", "ROUGH", "BETA", "NUC_FRAC"):
            g[k] = float(rng.uniform(lo[k], hi[k]))
        return g
    wp = cfg.WIDE_PRIOR
    if free_radius and rng.random() < wp["p_free_radius"]:
        g["RADIUS"] = _logu(rng, *wp["CELL_RADIUS"])
    else:
        # around the tile's size scale, so the spacing (drawn from the same scale) can actually
        # hold the cells -- a big cell between close seeds is squeezed to the seed spacing
        g["RADIUS"] = float(np.clip(_RADIUS_CENTRE * np.exp(wp["TYPE_RADIUS_SD"]
                                                            * rng.standard_normal()),
                                    wp["CELL_RADIUS"][0], hi["RADIUS"]))
    u = rng.random()
    if u < 0.05:
        g["ELONG"] = float(rng.uniform(lo["ELONG"], wp["ROUND_ELONG"][0]))
    elif u < 0.05 + wp["p_round"]:
        g["ELONG"] = float(rng.uniform(*wp["ROUND_ELONG"]))
    else:
        g["ELONG"] = _logu(rng, wp["ROUND_ELONG"][1], hi["ELONG"])
    g["ROUGH"] = float(rng.uniform(lo["ROUGH"], hi["ROUGH"]))
    g["BETA"] = float(rng.uniform(lo["BETA"], hi["BETA"]))
    r_nuc = _logu(rng, *wp["NUC_RADIUS_UM"]) / cfg.UM_PER_VOX
    g["NUC_FRAC"] = float(np.clip(r_nuc / g["RADIUS"], lo["NUC_FRAC"] + 0.02, hi["NUC_FRAC"] - 0.02))
    return g


def _logit(u):
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    return np.log(u / (1.0 - u))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _mutate_geometry(rng, g):
    """A child's geometry: the parent's, nudged. log RADIUS gets its own Gaussian; the other four
    are perturbed in logit space of their normalised value (so a scalar near a bound moves less and
    never leaves it) and clamped back to their bounds."""
    lo, hi = _geometry_bounds()
    out = dict(g)
    out["RADIUS"] = float(np.clip(g["RADIUS"] * np.exp(RADIUS_LOG_SD * rng.standard_normal()),
                                  lo["RADIUS"], hi["RADIUS"]))
    for k in ("ELONG", "ROUGH", "BETA", "NUC_FRAC"):
        u = float(np.clip(_geom_u(k, g[k], lo[k], hi[k]), 0.0, 1.0))
        u = _sigmoid(_logit(u) + GEOM_LOGIT_SD * rng.standard_normal())
        out[k] = float(_geom_from_u(k, u, lo[k], hi[k]))
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
    # p_pos may be one number (every marker alike, the run12 prior) or one per marker
    p_pos = np.broadcast_to(np.asarray(p_pos, float), (M,))

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
    for i_root in range(n_roots):
        for _try in range(max_retries):
            pos_row = rng.random(M) < p_pos
            pos_row[counterstain] = True
            pos_row[bg] = False                              # background-only markers: never expressed
            lvl_row = np.full(M, np.nan)
            lvl_row[pos_row] = [draw_level() for _ in range(int(pos_row.sum()))]
            geom = _sample_geometry(rng, free_radius=i_root > 0)
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
                 p_background=0.0, wide=None):
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

    `wide` (cfg.WIDE_PRIOR, or None for the run12 prior) adds: `ecm[m]`, an EXTRACELLULAR marker --
    collagen, a basement membrane, the capillary bed between muscle cells -- which no cell type
    expresses and which lights the matrix instead (flagged background too, for exactly that
    reason); and `p_pos[m]`, the odds that a root type is positive for marker m, so a panel holds
    ubiquitous markers next to rare ones instead of every marker at 0.25.
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
    out = dict(M=M, mu=mu, dye=dye, noise=noise, order=order, counterstain=0,
               background=np.array(background, bool), ecm=np.zeros(M, bool),
               p_pos=np.full(M, 0.25))
    if wide is None:
        return out
    # wide prior, drawn after everything above so the run12 stream is untouched: WHERE the markers
    # sit (a per-tile nuclear share instead of 42% of every panel), which are extracellular, and
    # how many types each is positive in
    q_nuc = float(rng.uniform(*wide["p_nuclear"]))
    q_ecm = float(rng.uniform(*wide["p_ecm"]))
    split = -0.1                                            # tau just inside the envelope
    for m in range(1, M):
        out["mu"][m] = (float(rng.uniform(MU_LO, split)) if rng.random() < q_nuc
                        else float(rng.uniform(split, MU_HI)))
        out["ecm"][m] = bool(rng.random() < q_ecm)
        out["p_pos"][m] = (float(rng.uniform(*wide["UBIQ_POS"])) if rng.random() < wide["p_ubiquitous"]
                           else float(rng.uniform(*wide["P_POS"])))
    out["background"] = out["background"] | out["ecm"]      # no cell type expresses a matrix marker
    return out


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


def build_marker(rng, *, name, dye, mu, noise_kind_idx, band_jitter=None):
    """One antibody channel. Its texture components are drawn from the existing noise priors (they
    are theta), but its tau localisation is set by the continuous `mu` (plus width/sharp tied to it
    via `band_from_mu`) on every component, so the marker shows the same way in every type that
    expresses it. `amp` is pinned to 1.0: in flexible mode the absolute brightness lives entirely in
    each type's expression level, so the old amp x level product (which could trade off invisibly)
    is gone.
    """
    comps = [NOISE_KINDS[k]() for k in noise_kind_idx]
    mu_v, width_v, sharp_v = band_from_mu(mu)
    if band_jitter is not None:
        # one draw per MARKER (every component shares the band): a membrane stain can be a hairline
        # or a fat cortical shell, a nuclear one flat-topped or centre-weighted. Not predicted --
        # mu stays the one localisation target.
        width_v = width_v * float(np.exp(band_jitter[0] * rng.standard_normal()))
        sharp_v = sharp_v * float(np.exp(band_jitter[1] * rng.standard_normal()))
    for c in comps:
        c.mu = narrow(c.mu, c.mu.lo, c.mu.hi, mu_v)
        c.width = narrow(c.width, c.width.lo, c.width.hi, width_v)
        c.sharp = narrow(c.sharp, c.sharp.lo, c.sharp.hi, sharp_v)
    m = PanelMarker(name=name, fluorophore=dye, noise_components=comps)
    m.amp = narrow(m.amp, 1.0, 1.0, 1.0)                     # pinned; brightness is in expression
    return m


def _flexible_architecture(rng, *, max_structures=3, wide=False):
    """A small tissue architecture for flexible mode: 0..max_structures bundles/vessels of random
    kind. ZERO is now reachable -- a structure-free, cell-only (or, with COVER=0, empty) tile, which
    build_architecture handles by uniform thinning with no director. Profiles are left empty --
    compositions (per kind) replace the per-instance log-weights, so no structure needs a Profile any
    more. FRAC/WALL_IN are narrowed and CONTACT pinned exactly as the fixed-mode builder does, for the
    same feasibility reasons (a lumen must not outgrow the tissue budget; the compliance solve must
    not engage).

    `wide`: each structure's share, size, aspect and DIRECTION are drawn too. They used to sit at
    the class defaults -- every structure 10% of the tissue, one size, running along +x -- so the
    ARCH.* targets were constants and the model had never seen a diagonal bundle. The direction is
    part of theta, and `targets_v3.augment_bundle` turns it with the image."""
    n = int(rng.integers(0, max_structures + 1))
    structures, kinds = [], []
    if n > 0:
        frac_hi = 0.9 / n
        for i in range(n):
            kind = "vessel" if rng.random() < 0.5 else "ordered"
            cls = Vasculature if kind == "vessel" else Ordered
            st = cls(name=f"s{i}", Profile={})
            hi = frac_hi * (0.5 if kind == "vessel" else 1.0)
            st.FRAC = narrow(st.FRAC, st.FRAC.lo, hi,
                             float(rng.uniform(min(0.04, hi), hi)) if wide else None)
            if kind == "vessel":
                st.WALL_IN = narrow(st.WALL_IN, st.WALL_IN.lo, min(st.WALL_IN.hi, 0.5))
            if wide:
                # the upper size grows with the field edge: on a 160 px tile (52 um) a bundle stops
                # at 40 um, on a 512 px region (166 um) at 128 um -- bundles and vessels at the scale
                # real tissue has them, which a 52 um frame could not hold. Capped by the class
                # bounds (vessel radius 35 um, bundle 600 um), so the theta normalisation is the
                # same; at the 160 px field the draw is the old one.
                grow = float(cfg.TISSUE_VOL[1]) / cfg.FIELD_REF
                lo_um, hi_um = (3.0, 20.0) if kind == "vessel" else (5.0, 40.0)
                hi_um = min(hi_um * max(grow, 1.0), st.SIZE_UM.hi)
                st.SIZE_UM = narrow(st.SIZE_UM, st.SIZE_UM.lo, st.SIZE_UM.hi, _logu(rng, lo_um, hi_um))
                st.ASPECT = narrow(st.ASPECT, st.ASPECT.lo, st.ASPECT.hi,
                                   float(rng.uniform(1.0, 5.0)))
                st.DIR_DEG = narrow(st.DIR_DEG, st.DIR_DEG.lo, st.DIR_DEG.hi,
                                    float(rng.uniform(-90.0, 90.0)))
            structures.append(st)
            kinds.append(kind)
    arch = TissueArchitecture(Structures=structures)
    arch.CONTACT = narrow(arch.CONTACT, 0.0, 0.0)
    return arch, kinds


def _expected_visible_cells(tg):
    """Rough estimate of how many cell profiles a tile of this density SHOWS. A section of
    thickness t through tissue packed in 3D at hard-core spacing d cuts every cell whose body
    reaches it: seed density x frame area x (t + the cells' own height). At Matern saturation the
    seed density is ~0.9 / (4/3 pi d^3) and a packed cell is ~1.3 d tall, so

        N ~ COVER x 0.9 x A x (t + 1.3 d) / (4.19 d^3)      (A = 160^2 px, t = 12 vox = 4 um)

    -- 26 at d = 20 vox, ~180 at the 9 vox floor. That raw count is then bent by a power law fitted
    to 340 non-empty wide-prior renders (2026-11-16): realised visible cells = 2.4 x raw^0.82
    (log-correlation 0.96; sparse tiles show about twice the raw count because elongated and grown
    cells reach the section from further away, dense ones slightly fewer because slivers are not
    counted as visible). Used to cap the requested type count (a present type needs several
    visible cells) and as the density stratum the rounds orchestrator steers (`density_stratum`).
    Deliberately approximate: it bounds a count, it is not a target."""
    md = max(float(tg.MIN_DIST.v), 1e-3)
    area = float(cfg.TISSUE_VOL[1] * cfg.TISSUE_VOL[2])
    t = float(cfg.OPTICS["section_um"]) / cfg.UM_PER_VOX
    raw = float(tg.COVER.v) * 0.9 * area * (t + 1.3 * md) / (4.19 * md ** 3)
    return _VISIBLE_CAL[0] * raw ** _VISIBLE_CAL[1]


# (a, b) of realised = a x raw^b, see _expected_visible_cells
_VISIBLE_CAL = (2.4, 0.82)
# visible-cell strata, the same edges as train_v3.DENSITY_BINS: empty, 1-10, 11-50, 51-100, > 100
DENSITY_EDGES = (0.5, 10.5, 50.5, 100.5)


def density_stratum(world):
    """Which density bin a drawn world is EXPECTED to land in (0 = empty .. 4 = > 100 cells),
    decided from theta alone, before anything is rendered -- so a generator can be asked for a
    given mix of densities by redrawing worlds, at no rendering cost."""
    if world["META"].get("empty"):
        return 0
    # per 160 px of FIELD: on a larger region the bins keep their meaning -- cells per 52 um x 52 um,
    # the window the instance head and its per-window metrics work in (train_v3.DENSITY_BINS)
    n = _expected_visible_cells(world["TG"]) * cfg.FIELD_REF ** 2 / float(
        cfg.TISSUE_VOL[1] * cfg.TISSUE_VOL[2])
    return 1 + int(np.searchsorted(DENSITY_EDGES[1:], n))


def _dapi_brightness(bright, b_min, b_max, centre=None):
    """Counterstain brightness from the theta level (drawn on the antibody range [b_min, b_max]).
    With a tile `centre` (wide prior) the level's position u in that range places the type within
    cfg.DAPI_TYPE_DECADES around the centre: same ordering across types, but the nuclei of one tile
    stay within about a decade of each other, as real ones do. Without one (centre None) u is
    mapped log-affinely onto the whole of cfg.DAPI_BRIGHT, the older rule."""
    lo, hi = cfg.DAPI_BRIGHT
    u = float(np.clip((np.log(max(bright, 1e-12)) - np.log(b_min))
                      / (np.log(b_max) - np.log(b_min) + 1e-12), 0.0, 1.0))
    if centre is None:
        return float(lo * (hi / lo) ** u)
    return float(np.clip(centre * 10.0 ** ((u - 0.5) * cfg.DAPI_TYPE_DECADES), lo, hi))


def _widen_component(rng, c, i=0):
    """One texture component drawn over its class bounds: every scalar in texture_keys it carries
    (uniform; they were never sampled, so all markers of one noise kind had ONE look), its strength
    `s` floored so a texture never vanishes outright, and `w` for the extra components (the first
    one's w stays pinned, PIN_DEGENERATE)."""
    wp = cfg.WIDE_PRIOR
    for key in wp["texture_keys"]:
        p = getattr(c, key, None)
        if isinstance(p, P):
            # a LENGTH (grain, period, fibre length, in um) is a scale: log-uniform, equal odds per
            # octave. Drawn uniformly over e.g. 0.1-3 um half of all textures were coarser than
            # 1.5 um and the optics left a smooth blur, where real stains are crisp at 1-2 px.
            v = (_logu(rng, p.lo, p.hi) if key in wp["texture_log_keys"] and p.lo > 0
                 else float(rng.uniform(p.lo, p.hi)))
            setattr(c, key, narrow(p, p.lo, p.hi, v))
    c.s = narrow(c.s, c.s.lo, c.s.hi, float(rng.uniform(max(wp["texture_s"][0], c.s.lo),
                                                         min(wp["texture_s"][1], c.s.hi))))
    if i > 0:
        c.w = narrow(c.w, c.w.lo, c.w.hi, float(rng.uniform(*wp["texture_w"])))


def _widen_textures(rng, panel):
    """Wide prior, per marker: sampled textures, cell-to-cell spread and (for a quarter of the
    antibody markers) polarity. The counterstain gets its own treatment -- its band and texture
    were fixed, one soft-edged nucleus for every tile (off/on contrast 0.22 against 0.03-0.35 on
    real stains): here the band runs from a flat-topped, crisp-edged fill to a peripheral ring and
    the chromatin from smooth to clumped, inside the bounds make_nuclear put on it, so whatever is
    drawn stays inside the nucleus."""
    wp = cfg.WIDE_PRIOR
    for mk in panel.Markers.values():
        dapi = mk.fluorophore == cfg.DAPI_DYE
        for i, c in enumerate(mk.noise_components):
            _widen_component(rng, c, i)
        if dapi:
            mu_v = float(rng.uniform(-0.75, -0.20))
            width_v = float(rng.uniform(0.35, 0.85))
            sharp_v = float(rng.uniform(2.5, 10.0))
            for c in mk.noise_components:
                c.mu = narrow(c.mu, c.mu.lo, c.mu.hi, mu_v)
                c.width = narrow(c.width, c.width.lo, c.width.hi, width_v)
                c.sharp = narrow(c.sharp, c.sharp.lo, c.sharp.hi, sharp_v)
            cv = float(rng.uniform(*wp["DAPI_CELL_CV"]))
        else:
            cv = float(rng.uniform(*wp["CELL_CV"]))
            if rng.random() < wp["p_polar"]:
                mk.polarity = narrow(mk.polarity, mk.polarity.lo, mk.polarity.hi,
                                     float(rng.uniform(*wp["POLARITY"])))
                v = rng.standard_normal(3)
                v = v / (np.linalg.norm(v) + 1e-12)
                mk.pol_dir = tuple(narrow(pd, pd.lo, pd.hi, float(x))
                                   for pd, x in zip(mk.pol_dir, v))
        mk.cell_cv = narrow(mk.cell_cv, mk.cell_cv.lo, mk.cell_cv.hi, cv)


def sample_background(rng, panel, panel_spec, names_canon, type_names, n_noise_kinds):
    """The tile's SHARED signal: what every channel shows whether or not its antibody binds.

    On a real staged slide about half the channels are the same picture at different brightness --
    residual autofluorescence and nonspecific binding -- and that picture is the tissue's own fine
    structure (fibrils, organelles, gaps), not a blurred outline of the cells. The old generator
    used the cell occupancy times one smooth texture, so EVERY synthetic channel traced the cells
    as blobs on black, and "a blob in most channels = a cell" was a perfect shortcut that real
    tissue never honours. Here the shared signal has two parts:

    - `markers`: 1-3 hidden patterns rendered PER CELL exactly like an antibody marker (their own
      localisation band, broad; their own textures, oriented with the cell), at a level that
      differs per TYPE (`type_level`, lognormal: some cell types are simply more autofluorescent)
      and per cell (`cell_cv`). Channel m shows them mixed by `mix[m]` at its own
      PanelMarker.tissue_bg level -- so unstained channels look alike but not identical.
    - `matrix`: 1-3 extracellular textures filling the tissue the cells do NOT claim, optionally
      with a pericellular band `peri` hugging every cell boundary (a basement membrane, the
      capillary bed between muscle cells). Channel m shows them at `matrix_level[m]`: faint for an
      ordinary channel (MATRIX_REL x its cell background), bright for an EXTRACELLULAR marker
      (`panel_spec["ecm"]`), which no cell type expresses.

    None of it is theta: nuisance the model has to see through. Keyed by marker NAME so the
    channel permutation cannot misalign it."""
    from parameter import ClusterNoise, FibreNoise, BlobNoise, NetworkNoise
    wp = cfg.WIDE_PRIOR
    M = panel_spec["M"]
    n_bg = int(rng.integers(wp["n_bg"][0], wp["n_bg"][1] + 1))
    markers, cell_cv = [], []
    for j in range(n_bg):
        k = int(rng.integers(1, 3))
        mk = build_marker(rng, name=f"__bg{j}", dye=cfg.DYES[0], mu=float(rng.uniform(MU_LO, MU_HI)),
                          noise_kind_idx=tuple(int(x) for x in rng.integers(0, n_noise_kinds, size=k)))
        width_v, sharp_v = float(rng.uniform(0.6, 1.5)), float(rng.uniform(1.0, 4.0))
        for i, c in enumerate(mk.noise_components):
            _widen_component(rng, c, i)
            # broad and soft: the shared signal fills the cell, it does not pick a compartment;
            # and a mild relief, not the hot blobs a strong antibody texture makes -- a cluster
            # gate at fill 0.02 is a few specks at 50x the mean, so the gate stays mostly open
            c.width = narrow(c.width, c.width.lo, c.width.hi, width_v)
            c.sharp = narrow(c.sharp, c.sharp.lo, c.sharp.hi, sharp_v)
            c.s = narrow(c.s, c.s.lo, c.s.hi, float(rng.uniform(*wp["BG_TEXTURE_S"])))
            if isinstance(getattr(c, "fill", None), P):
                c.fill = narrow(c.fill, c.fill.lo, c.fill.hi, float(rng.uniform(0.3, c.fill.hi)))
        markers.append(mk)
        cell_cv.append(float(rng.uniform(*wp["BG_CELL_CV"])))
    sd = float(wp["BG_TYPE_SD"])
    type_level = {t: np.exp(sd * rng.standard_normal(n_bg) - 0.5 * sd * sd) for t in type_names}
    mix = {nm: rng.dirichlet(np.full(n_bg, 0.7)) for nm in names_canon}

    n_mat = int(rng.integers(wp["n_matrix"][0], wp["n_matrix"][1] + 1))
    kinds = (ClusterNoise, FibreNoise, BlobNoise, NetworkNoise)
    matrix = []
    for j in range(n_mat):
        comp = kinds[int(rng.integers(len(kinds)))]()
        _widen_component(rng, comp, 0)
        matrix.append(dict(comp=comp, azim=float(rng.uniform(0.0, 180.0)),
                           peri=(float(rng.uniform(*wp["PERI"])) if rng.random() < wp["p_peri"] else 0.0),
                           peri_um=_logu(rng, *wp["PERI_UM"])))
    matrix_level = {}
    for m, nm in enumerate(names_canon):
        lv = np.zeros(n_mat)
        if panel_spec["ecm"][m]:
            lv[int(rng.integers(n_mat))] = _logu(rng, *wp["ECM_LEVEL"])
        else:
            lv = float(panel.Markers[nm].tissue_bg.v) * _logu(rng, *wp["MATRIX_REL"]) \
                * rng.dirichlet(np.full(n_mat, 0.7))
        matrix_level[nm] = lv
    return dict(markers=markers, cell_cv=cell_cv, type_level=type_level, mix=mix,
                matrix=matrix, matrix_level=matrix_level)


def _widen_detector(rng, det):
    """Wide prior: the camera and staining run drawn per tile within cfg.WIDE_PRIOR["DET"] (each
    inside the Detector class bounds). Returns a NEW Detector -- the caller's stays untouched."""
    import copy
    det = copy.deepcopy(det)
    for key, (lo, hi) in cfg.WIDE_PRIOR["DET"].items():
        p = getattr(det, key)
        setattr(det, key, narrow(p, p.lo, p.hi, float(rng.uniform(lo, hi))))
    return det


def build_flexible_world(rng, *, curriculum=None, dyes, n_noise_kinds=len(NOISE_KINDS),
                         dapi_name="DAPI", max_structures=3, detector=None, prior=None):
    """Draw a whole flexible-mode sample: the panel, the lineage cell-type set, the tissue
    architecture and the per-context compositions, assembled into the same world dict that
    `render_frame` already consumes (PANEL / TYPES / ARCH / CELL / TG / FRACTIONS), plus the extra
    flexible-mode fields COMPOSITIONS, PANEL_SPEC, TYPESET and META that carry the sample's layout
    for theta v3 and the data loader.

    `curriculum` is the plan 3.7 knob dict (m_range, k_true_range, max_depth, min_level_diff);
    None uses the final prior. The order in which markers appear in the panel (and so in the image)
    is a per-sample permutation, so channel position carries no information.

    `prior` ("wide" / "run12", default cfg.PRIOR) picks the prior -- see config.WIDE_PRIOR. "run12"
    reproduces the run12 training distribution bit-for-bit for a given rng seed.
    """
    global _ELONG_OVERRIDE, _RADIUS_CENTRE
    prior = prior or cfg.PRIOR
    if prior not in ("wide", "run12"):
        raise ValueError(f"prior {prior!r} unknown; use 'wide' (cfg.WIDE_PRIOR) or 'run12'.")
    wide = prior == "wide"
    _ELONG_OVERRIDE = None if wide else cfg.RUN12_PRIOR["ELONG"]
    try:
        return _build_flexible_world(rng, curriculum=curriculum, dyes=dyes,
                                     n_noise_kinds=n_noise_kinds, dapi_name=dapi_name,
                                     max_structures=max_structures, detector=detector, wide=wide)
    finally:
        _ELONG_OVERRIDE = None
        _RADIUS_CENTRE = None


def _build_flexible_world(rng, *, curriculum, dyes, n_noise_kinds, dapi_name, max_structures,
                          detector, wide):
    from parameter import TissueGeometry, Artifacts, narrow as _narrow   # local: heavy import
    import config as _cfg
    cur = dict(m_range=(8, 40), k_true_range=(1, 30), max_depth=4, min_level_diff=0.3,
               p_background=0.0)
    cur.update(curriculum or {})
    wp = _cfg.WIDE_PRIOR
    if wide:
        cur["m_range"] = wp["m_range"]
        cur["p_background"] = float(rng.uniform(*wp["p_background"]))

    # --- per-tile tissue density, drawn BEFORE the type set so the type count can be capped to what
    # the cellularity can actually show. The model must handle the whole span (dense-no-background ->
    # empty) at every round, so density is NOT interpolated by the curriculum. An empty tile is
    # COVER=0 (support_mask returns all-False) with zero structures -- a genuine zero-cell negative. ---
    dens = dict(_cfg.DENSITY)
    if wide:
        dens.update({k: wp[k] for k in ("MIN_DIST", "COVER", "GROW")})
    empty = bool(rng.random() < float(dens.get("p_empty", 0.0)))
    ms = 0 if empty else int(dens["max_structures"])
    arch, kinds = _flexible_architecture(rng, max_structures=ms, wide=wide)

    tg = TissueGeometry()
    confluent = False
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
        if wide:
            # one cell-size scale per tile, and spacing FOLLOWS it (drawn on its own the spacing
            # squeezed every cell to ~3 um whatever its RADIUS). Whether the cells then fill the
            # tissue is a REGIME, not an accident of two independent draws -- see
            # config.WIDE_PRIOR["p_confluent"].
            global _RADIUS_CENTRE
            _RADIUS_CENTRE = _logu(rng, *wp["CELL_RADIUS"])
            confluent = bool(rng.random() < wp["p_confluent"])
            if confluent:
                ratio = _logu(rng, *wp["CONFLUENT_RATIO"])
                grow = float(rng.uniform(min(wp["GROW_FILL"] * ratio, tg.GROW.hi), tg.GROW.hi))
                cov = (cov_hi if rng.random() < wp["p_full_cover"]
                       else float(max(rng.uniform(0.3, cov_hi), rng.uniform(0.3, cov_hi))))
            else:
                ratio = _logu(rng, *wp["SPACING_RATIO"])
                grow = float(rng.uniform(*dens["GROW"]))
                cov = float(max(rng.uniform(cov_lo, cov_hi), rng.uniform(cov_lo, cov_hi)))
            md = float(np.clip(_RADIUS_CENTRE * ratio, max(wp["MIN_SPACING"], tg.MIN_DIST.lo),
                               tg.MIN_DIST.hi))
        else:
            md = float(rng.uniform(*dens["MIN_DIST"]))
            cov = float(rng.uniform(cov_lo, cov_hi))
            grow = float(rng.uniform(*dens["GROW"]))
            confluent = False
        tg.MIN_DIST = _narrow(tg.MIN_DIST, tg.MIN_DIST.lo, tg.MIN_DIST.hi, md)
        tg.COVER = _narrow(tg.COVER, tg.COVER.lo, tg.COVER.hi, cov)
        tg.GROW = _narrow(tg.GROW, tg.GROW.lo, tg.GROW.hi, grow)

    # cap the requested number of distinct types to what this cellularity can realise: a type counts
    # as present only with >= VIS.min_visible_cells visible cells, so a sparse tile that asked for many
    # types would end up with ZERO present types (no typing signal) and bias the existence head toward
    # "no types". Dense tiles are unaffected (the cap exceeds the curriculum ceiling there).
    klo, khi = cur["k_true_range"]
    k_cap = max(1, int(round(_expected_visible_cells(tg) / float(_cfg.VIS["min_visible_cells"]))))
    k_range = (min(klo, k_cap), min(khi, k_cap))

    panel_spec = sample_panel(rng, m_range=cur["m_range"], dyes=dyes, n_noise_kinds=n_noise_kinds,
                              p_background=cur["p_background"], wide=wp if wide else None)
    M = panel_spec["M"]
    bg = panel_spec["background"]
    ts = sample_typeset(rng, M, k_true_range=k_range, max_depth=cur["max_depth"],
                        min_level_diff=cur["min_level_diff"], counterstain=panel_spec["counterstain"],
                        background=bg, p_pos=panel_spec["p_pos"] if wide else 0.25)
    K = ts.K
    b_min, b_max = expression_range()

    # markers in CANONICAL order, then inserted into the panel in the permuted channel order
    canon = {}
    for m in range(M):
        nm = _marker_name(m, dapi_name)
        if m == panel_spec["counterstain"]:
            # the counterstain, always nuclear; under the wide prior its chromatin texture may be
            # nearly smooth (strength down to 0.1), which the 0.6 floor ruled out
            canon[nm] = dapi_marker(name=nm, strength=(0.1, 1.6)) if wide else dapi_marker(name=nm)
        else:
            mk = build_marker(rng, name=nm, dye=panel_spec["dye"][m], mu=panel_spec["mu"][m],
                              noise_kind_idx=panel_spec["noise"][m],
                              band_jitter=((wp["BAND_WIDTH_SD"], wp["BAND_SHARP_SD"]) if wide
                                           else None))
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
    g0 = CellGeometry()
    # the tile's counterstain brightness (one staining run, one exposure): types sit around it
    d_lo, d_hi = _cfg.DAPI_BRIGHT
    half = 10.0 ** (0.5 * _cfg.DAPI_TYPE_DECADES)
    dapi_centre = _logu(rng, d_lo * half, d_hi / half) if wide else None
    for k in range(K):
        gk = {key: float(ts.geometry[k, i]) for i, key in enumerate(GEOM_KEYS)}
        if wide:
            # shape NUISANCE per type, outside the five predicted scalars: the nucleus's own
            # aspect / lobing / offset (it was one shape for every type: NUC_ROUGH at its maximum,
            # NUC_ELONG 1.6) and how far the type follows the tissue grain (ALIGN was never
            # sampled, so no cell ever did)
            gk.update(
                NUC_ELONG=(1.0 if rng.random() < wp["p_nuc_round"] else _logu(rng, *wp["NUC_ELONG"])),
                NUC_ROUGH=float(rng.uniform(*wp["NUC_ROUGH"])),
                NUC_BETA=float(rng.uniform(*wp["NUC_BETA"])),
                NUC_CORR=float(rng.uniform(g0.NUC_CORR.lo, g0.NUC_CORR.hi)),
                NUC_OFFSET=float(rng.uniform(g0.NUC_OFFSET.lo, g0.NUC_OFFSET.hi)),
                RIM=float(rng.uniform(*wp["RIM"])),
                ALIGN=(1.0 if rng.random() < wp["p_type_aligned"] else float(rng.uniform(0.0, 1.0))))
        geom = CellGeometry(**gk)
        expr = {}
        for m in np.flatnonzero(ts.positive[k]):
            nm = names_canon[m]
            bright = float(10.0 ** ts.level[k, m])
            if wide and m == panel_spec["counterstain"]:
                bright = _dapi_brightness(bright, b_min, b_max, centre=dapi_centre)
            expr[nm] = P(0.0, max(b_max, bright), 0.05, bright, f"expr {nm}",
                         comment="absolute brightness of this marker in this type (flexible mode); "
                                 "replaces the panel amp x level product.")
        types[_type_name(k)] = CellType(name=_type_name(k), Color="black",
                                        Geometry=geom, Expression=expr)

    contexts, allowed, pi = sample_compositions(rng, kinds, K)
    struct_to_ctx = [contexts.index(kd) for kd in kinds]

    # wide-prior extras, drawn LAST so every earlier draw is the run12 stream. Tissue background
    # per channel (log-uniform), sampled textures, and a per-tile detector / staining run.
    if wide:
        for m in range(M):
            mk = canon[names_canon[m]]
            if m == panel_spec["counterstain"]:
                # a fraction of the stain itself, so the nuclei are never simply buried: from a
                # clean re-stain to the cycle-1 stain that is mostly blue autofluorescence
                level = dapi_centre * _logu(rng, *_cfg.DAPI_CONTRAST)
            else:
                lo, hi = _cfg.TISSUE_BG["marker"]
                level = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            mk.tissue_bg = _narrow(mk.tissue_bg, mk.tissue_bg.lo, mk.tissue_bg.hi, level)
        _widen_textures(rng, panel)
        if detector is not None:
            detector = _widen_detector(rng, detector)
        if not empty:
            # the section is cut from bulk: seed the slab as deep as the biggest type can reach
            reach = max(float(g.Geometry.RADIUS.v) * float(max(g.Geometry.ELONG.v, 1.0)) ** (2.0 / 3.0)
                        for g in types.values())
            block = float(_cfg.TISSUE_VOL[0] * _cfg.Z_RATIO)
            slab = float(np.clip(block + 2.0 * wp["SLAB_REACH"] * reach, tg.SLAB.lo,
                                 min(wp["SLAB_MAX"], tg.SLAB.hi)))
            tg.SLAB = _narrow(tg.SLAB, tg.SLAB.lo, tg.SLAB.hi, slab)
            couple = float(rng.uniform(*wp["SIZE_COUPLE"]))
            # the smallest type's spacing is MIN_DIST x (R_min / R_mean)^couple: keep it above the
            # spacing floor, or a small type between big ones packs past the query cap
            lr = np.log([float(g.Geometry.RADIUS.v) for g in types.values()])
            d_lr = float(lr.min() - lr.mean())
            room = np.log(wp["MIN_SPACING"] / float(tg.MIN_DIST.v))
            if d_lr < -1e-6 and room > d_lr * couple:
                couple = float(np.clip(0.9 * room / d_lr, 0.0, couple))
            tg.SIZE_COUPLE = _narrow(tg.SIZE_COUPLE, tg.SIZE_COUPLE.lo, tg.SIZE_COUPLE.hi, couple)
            if rng.random() < wp["p_axis"]:
                tg.AXIS_ALIGN = _narrow(tg.AXIS_ALIGN, tg.AXIS_ALIGN.lo, tg.AXIS_ALIGN.hi,
                                        float(rng.uniform(*wp["AXIS_ALIGN"])))
                tg.AXIS_AZIM_DEG = _narrow(tg.AXIS_AZIM_DEG, tg.AXIS_AZIM_DEG.lo,
                                           tg.AXIS_AZIM_DEG.hi, float(rng.uniform(0.0, 180.0)))
                # isotropic over the half-sphere: cos(polar) uniform, so in-plane grain (a
                # lengthwise cut) is commoner than a grain running straight through the section
                tg.AXIS_POLAR_DEG = _narrow(tg.AXIS_POLAR_DEG, tg.AXIS_POLAR_DEG.lo,
                                            tg.AXIS_POLAR_DEG.hi,
                                            float(np.rad2deg(np.arccos(rng.uniform(0.0, 1.0)))))
    elif detector is not None:
        # run12 rendered DAPI on the camera pedestal, before the staged offset correction existed
        import copy
        detector = copy.deepcopy(detector)
        detector.DAPI_OFFSET_ADU = _narrow(detector.DAPI_OFFSET_ADU, detector.DAPI_OFFSET_ADU.lo,
                                           detector.DAPI_OFFSET_ADU.hi, float(detector.OFFSET_ADU.v))
    compositions = dict(contexts=contexts, pi=pi, allowed=allowed,
                        struct_to_ctx=struct_to_ctx, type_order=[_type_name(k) for k in range(K)])
    background, optics = None, None
    if wide:
        background = sample_background(rng, panel, panel_spec, names_canon, list(types),
                                       n_noise_kinds)
        optics = dict(na=float(wp["NA"][int(rng.integers(len(wp["NA"])))]),
                      focal_um=float(wp["FOCUS_UM"][int(rng.integers(len(wp["FOCUS_UM"])))]))

    world = {
        "PANEL": panel, "TYPES": types, "TG": tg, "DET": detector,
        "ARCH": arch, "CELL": CellGeometry(), "FRACTIONS": list(np.ones(K) / K),
        "COMPOSITIONS": compositions, "PANEL_SPEC": panel_spec, "TYPESET": ts,
        # artifact parameters (fibres, aggregates, detachment, cycle shift): only read when the frame
        # is rendered with_artifacts, which the generator decides per sample
        "AR": Artifacts(),
        # the shared background (sample_background) and the per-tile optics (NA, focus): both
        # nuisance, both None under the run12 prior, where render_frame falls back to the old
        # occupancy background and the default objective
        "BG": background, "OPT": optics,
        "META": dict(K=K, M=M, counterstain=panel_spec["counterstain"],
                     channel_order=order, b_range=(b_min, b_max),
                     mu=list(panel_spec["mu"]),                 # canonical-order localisation targets
                     empty=empty, n_struct=len(arch.Structures), confluent=bool(confluent)),
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
    """Normalise the [K, 5] geometry block column-by-column over the sampling bounds: RADIUS and
    ELONG in log (they are scales), the rest linear. `geometry_from_u` is the inverse."""
    lo, hi = _geometry_bounds()
    out = np.zeros_like(geometry, np.float64)
    for i, k in enumerate(GEOM_KEYS):
        out[:, i] = _geom_u(k, geometry[:, i], lo[k], hi[k])
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

    # NUISANCE the model never predicts, stored only so a tile can be put in a stratum afterwards
    # (which objective, in focus or not, confluent or loose, how deep the slab): the per-stratum
    # error tables and the real-vs-synthetic coverage checks read these
    opt = world.get("OPT") or {}
    nuis = dict(na=float(opt.get("na", cfg.OPTICS["na"])), focal_um=float(opt.get("focal_um", 0.0)),
                confluent=float(meta.get("confluent", False)), empty=float(meta.get("empty", False)),
                slab=float(tgw.SLAB.v), size_couple=float(tgw.SIZE_COUPLE.v),
                axis_align=float(tgw.AXIS_ALIGN.v), min_dist=float(tgw.MIN_DIST.v),
                grow=float(tgw.GROW.v), cover=float(tgw.COVER.v),
                n_struct=float(meta.get("n_struct", 0)),
                expected_cells=float(_expected_visible_cells(tgw)),
                stratum=float(density_stratum(world)))

    return {
        "nuis_names": np.array(list(nuis)),
        "nuis": np.array(list(nuis.values()), np.float64),
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
