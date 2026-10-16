"""ParamSpace: the fixed, composition-scoped addressing of theta for one round-based run.

Phase 1 (`composition_mode: single`) fixes ONE composition -- one `panel_structure`/
`architecture_structure` draw -- for the whole run, so a `ParamSpace` built from that
composition's reference world has a fixed path list, and every sample's theta uses that exact
layout: validity is trivially all-True. Phase 2 (`composition_mode: full`) will need a
fixed-CAPACITY layout with real per-sample masking; `ParamSpace`'s shape here (`validity_mask`
always returned, even though trivial today) is deliberately built so Phase 2 only needs a new
`validity_mask` body, not a new class.

Prior: Option A (agreed in planning) -- the prior equals the hard bounds, for every parameter,
exactly matching `parameter.sample_prior`'s existing behaviour. Expressed in normalised units,
that means every path's prior marginal is Uniform(0, 1), so `prior_var` is EXACTLY 1/12 for every
path, with no dependence on `tf` or the bound width -- a closed form, not an approximation.

`parameter.py` (and everything it drags in -- scipy, ipywidgets, render.py) is only ever needed
by the three methods that actually touch a live world tree (`from_reference_world`, `normalize`,
`set_vector`, `sample_prior`) -- those imports are deliberately LAZY (function-local), so that
`ParamSpace.load()`/`denormalize()`/`validity_mask()`/`log_prior()`/`prior_var()` (pure bounds/tf
arithmetic, or serialisation) work under the `synmodel` env too, which has neither scipy nor
ipywidgets. Training/evaluation only ever need that second group.
"""
import _pathfix  # noqa: F401  -- must run before the src/-level imports below

import json

import numpy as np

# Var(Uniform(0, 1)) -- the exact prior_var for every path under Option A (prior == hard bounds,
# expressed directly on the normalised u-axis that sample_prior/to_u/from_u already operate in).
PRIOR_VAR_UNIFORM01 = 1.0 / 12.0


class ParamSpace:
    """Fixed addressing of theta for one composition, built once per run from a reference world.

    `paths` is the sorted, deterministic path list `fitted_paths` returns for that composition.
    `groups[i]` is `path_role(paths[i])` -- a composition-independent identity for each parameter
    KIND (e.g. `PANEL.Markers.*.amp`), used to key curriculum/evaluation statistics so they stay
    meaningful if a later phase varies composition (Phase 1 never does, but the addressing is
    built so Phase 2 doesn't need a new statistics schema).
    """

    def __init__(self, paths, groups, lo, hi, tf):
        self.paths = list(paths)
        self.groups = list(groups)
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.tf = list(tf)
        n = len(self.paths)
        if not (len(self.groups) == n and len(self.lo) == n and len(self.hi) == n
                and len(self.tf) == n):
            raise ValueError(
                f"ParamSpace arrays must all have length {n} (len(paths)): "
                f"groups={len(self.groups)} lo={len(self.lo)} hi={len(self.hi)} "
                f"tf={len(self.tf)}"
            )
        self._index = {p: i for i, p in enumerate(self.paths)}

    @classmethod
    def from_reference_world(cls, root, pin=None, keep=None):
        """Build from a world tree with theta already declared (values don't matter -- only
        bounds/tf are read). This is the ONLY place `fitted_paths`/`bounds` get called for a
        round-based run: every later stage addresses theta purely through this `ParamSpace`.
        Generation-side only -- imports `parameter` lazily (see module docstring)."""
        from parameter import DEFAULT_PIN, bounds as _bounds, fitted_paths, get_p, path_role
        pin = DEFAULT_PIN if pin is None else pin
        paths = fitted_paths(root, pin=pin, keep=keep)
        if not paths:
            raise ValueError("fitted_paths returned no parameters for this composition/pin/keep")
        lo, hi = _bounds(root, paths)
        tf = [get_p(root, p).tf for p in paths]
        groups = [path_role(p) for p in paths]
        return cls(paths, groups, lo, hi, tf)

    def __len__(self):
        return len(self.paths)

    def index_of(self, path):
        return self._index[path]

    # -- normalisation -----------------------------------------------------------------------

    def normalize(self, root):
        """A world tree (theta already written in) -> u in [0, 1]^P, via `to_u` per path.
        Generation-side only -- imports `parameter` lazily (see module docstring)."""
        from parameter import get_p, to_u
        return np.array([to_u(get_p(root, p)) for p in self.paths], dtype=np.float64)

    def denormalize(self, u):
        """u in [0, 1]^P -> raw physical values, by bounds/tf alone (no world tree needed) --
        for manifests, reports and plots. To write u back into an actual world object, use
        `set_vector(root, u)` below."""
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        out = np.empty_like(u)
        for i in range(len(self.paths)):
            lo, hi = self.lo[i], self.hi[i]
            if self.tf[i] == "log" and lo > 0 and hi > 0:
                out[i] = np.exp(np.log(lo) + u[i] * (np.log(hi) - np.log(lo)))
            else:
                out[i] = lo + u[i] * (hi - lo)
        return out

    def set_vector(self, root, u):
        """Write u (normalised, [0, 1]^P) into `root` in place.
        Generation-side only -- imports `parameter` lazily (see module docstring)."""
        from parameter import set_vector as _set_vector
        return _set_vector(root, self.paths, u)

    # -- prior (Option A: prior == hard bounds, in normalised units) -------------------------

    def sample_prior(self, rng):
        """u ~ prior, in normalised units -- ParamSpace's own entry point to the SAME uniform
        draw `parameter.sample_prior` already does; not a new distribution.
        Generation-side only -- imports `parameter` lazily (see module docstring)."""
        from parameter import sample_prior as _sample_prior
        return _sample_prior(rng, self.paths)

    def log_prior(self, u):
        """log p(u) for the whole vector: uniform density 1 on [0, 1]^P under Option A, so
        log-density is 0 wherever every entry is in-range and -inf otherwise."""
        u = np.asarray(u, dtype=float)
        return 0.0 if np.all((u >= 0.0) & (u <= 1.0)) else -np.inf

    def log_prior_marginal(self, k, u_k):
        """log p_k(u_k), the single-dimension marginal `proposal.py`'s per-path density ratio
        (q_c(u)/p(u) = g(u_k)/p_k(u_k)) needs. Uniform density 1 on [0, 1] for every k under
        Option A -- note this means the ratio collapses to plain g(u_k), which `proposal.py`
        exploits directly rather than dividing by a value that's always 1."""
        return 0.0 if 0.0 <= float(u_k) <= 1.0 else -np.inf

    def prior_var(self, k=None):
        """Var_p[u_k] in normalised units. Exactly 1/12 for every k under Option A -- a closed
        form, not sampled. `k` is accepted for API symmetry with per-path callers; the value
        never actually varies while the prior is uniform on [0, 1]."""
        return PRIOR_VAR_UNIFORM01

    # -- validity (Phase 1: trivially all-True; Phase 2 will give this a real body) -----------

    def marker_field_map(self, n_markers):
        """Precomputed reassembly table for `model.SetParamNet`: for the fixed canonical channel
        order `panel_structure` always uses -- `[DAPI, M00, M01, ...]`, a pure function of
        `n_markers` alone, independent of struct_seed (only the DYE/noise-kind draws are
        seed-dependent, never the marker NAMES/order) -- which absolute path index each (marker
        slot, per-marker field) pair maps to, and which absolute path indices are global
        (non-marker-specific).

        Returns a dict with `marker_scatter` (`(n_markers, n_field)` int array, -1 where that
        field doesn't apply to that marker), `global_path_idx` (`(n_global,)` int array), and
        `field_vocab` (the field-identity strings, for debugging/logging).

        NOTE: unlike `self.groups` (`path_role`), list indices are kept LITERAL here (e.g.
        `noise_components[0].scale` and `noise_components[1].scale` are different vocabulary
        entries) -- a marker can have up to `comps[1]` (see `panel_structure`) independently
        parameterised noise components, so collapsing the index the way `path_role` does for
        cross-marker reporting would silently collide several distinct real paths onto the same
        output slot and drop the rest.
        """
        from config import DAPI_NAME
        marker_order = [DAPI_NAME] + [f"M{i:02d}" for i in range(n_markers - 1)]
        marker_slot = {name: c for c, name in enumerate(marker_order)}

        field_vocab = {}
        marker_path_idx = {}
        global_path_idx = []
        for i, path in enumerate(self.paths):
            parts = path.split(".")
            if len(parts) >= 3 and parts[0] == "PANEL" and parts[1] == "Markers":
                name = parts[2]
                if name not in marker_slot:
                    raise ValueError(
                        f"path {path!r} references marker {name!r}, not in the canonical "
                        f"marker order for n_markers={n_markers}: {marker_order}"
                    )
                field_parts = list(parts)
                field_parts[2] = "*"  # unify the field vocabulary across every marker (incl. DAPI)
                role = ".".join(field_parts)
                k = field_vocab.setdefault(role, len(field_vocab))
                marker_path_idx[(marker_slot[name], k)] = i
            else:
                global_path_idx.append(i)

        n_field = len(field_vocab)
        marker_scatter = -np.ones((len(marker_order), n_field), dtype=np.int64)
        for (c, k), i in marker_path_idx.items():
            marker_scatter[c, k] = i
        return dict(marker_scatter=marker_scatter,
                    global_path_idx=np.array(global_path_idx, dtype=np.int64),
                    field_vocab=list(field_vocab.keys()))

    # -- validity (Phase 1: trivially all-True; Phase 2 will give this a real body) -----------

    def validity_mask(self, composition=None):
        """Which of `self.paths` are valid theta entries for `composition`.

        Phase 1 uses exactly one composition for the whole run -- the one this ParamSpace was
        built from -- so every path is valid for every sample, always. `composition` is accepted
        (and ignored) so callers written against the eventual Phase 2 signature (validity varies
        by composition) don't need to change when Phase 2 introduces real masking.
        """
        return np.ones(len(self.paths), dtype=bool)

    # -- serialisation -------------------------------------------------------------------------

    def to_dict(self):
        return dict(paths=self.paths, groups=self.groups, lo=self.lo.tolist(),
                    hi=self.hi.tolist(), tf=self.tf)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def from_dict(cls, d):
        return cls(d["paths"], d["groups"], d["lo"], d["hi"], d["tf"])

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
