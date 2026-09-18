"""Phase 1 (`composition_mode: single`): one panel_structure/architecture_structure draw, fixed
for the whole run. Every rounds/*.py stage that needs a world/ParamSpace goes through here, so
they all build the exact same composition from the same small config dict rather than each
re-deriving it slightly differently.

`max_structures` is NOT configurable here: `gen_dataset.panel_structure` calls
`architecture_structure` internally with its own default (3), and plumbing a override through
would be a gen_dataset.py signature change beyond what this phase needs -- Phase 1 accepts that
default. Noted as a known limitation, not silently worked around.

`CompositionConfig` itself is a plain dataclass (json in/out only) so it stays importable under
`synmodel` (no scipy/numba/gen_dataset there); the builder functions below need the full
generation stack and import it lazily, function-local, exactly like `param_space.py` does for
`parameter.py` -- training/evaluation only ever need `CompositionConfig`, never these.
"""
import _pathfix  # noqa: F401

import dataclasses
import json


@dataclasses.dataclass
class CompositionConfig:
    n_markers: int
    n_types: int
    expr_frac: float = 0.6
    struct_seed: int = 0
    with_artifacts: bool = True
    max_aggregates: float = None
    max_fussels: float = None
    max_detach_cover: float = None

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self):
        return dataclasses.asdict(self)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


def build_struct(comp: CompositionConfig):
    """The discrete composition (dyes, noise kinds, cell-type support, architecture) -- a pure
    function of `comp`, independent of any particular sample's theta values.
    Generation-side only -- imports the generation stack lazily (see module docstring)."""
    from gen_dataset import panel_structure
    return panel_structure(comp.n_markers, comp.n_types, comp.expr_frac,
                           struct_seed=comp.struct_seed)


def build_reference_world(comp: CompositionConfig, struct=None):
    """A world tree with theta DECLARED (bounds/tf real) but not yet drawn to any particular
    sample's values -- everything `ParamSpace.from_reference_world` needs, nothing more. Matches
    gen_dataset.make_dataset's own reference-world draw (same master-seeded rng, not a literal).
    Generation-side only -- imports the generation stack lazily (see module docstring)."""
    import numpy as np
    from gen_dataset import generate_world
    struct = struct if struct is not None else build_struct(comp)
    return generate_world(struct, rng=np.random.default_rng(comp.struct_seed),
                          with_artifacts=comp.with_artifacts, max_aggregates=comp.max_aggregates,
                          max_fussels=comp.max_fussels, max_detach_cover=comp.max_detach_cover)


def build_param_space(comp: CompositionConfig, struct=None, keep=None):
    """Generation-side only -- imports the generation stack lazily (see module docstring)."""
    from gen_dataset import ARCH_PIN
    from parameter import DEFAULT_PIN
    import config as cfg
    from param_space import ParamSpace
    ref = build_reference_world(comp, struct=struct)
    return ParamSpace.from_reference_world(ref, pin=DEFAULT_PIN + ARCH_PIN,
                                           keep=keep if keep is not None else cfg.KEEP)
