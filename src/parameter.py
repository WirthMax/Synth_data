import re
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from render import _filtered
import numpy as np
import ipywidgets as W
from scipy.special import ndtri 

import config as cfg

PIN_DEGENERATE = cfg.PIN_DEGENERATE
PIN_GROUNDED = cfg.PIN_GROUNDED
PIN = cfg.PIN
KEEP = cfg.KEEP
DAPI_NAME = cfg.DAPI_NAME
DAPI_DYE = cfg.DAPI_DYE
NUCLEAR_TAU_MAX = cfg.NUCLEAR_TAU_MAX

@dataclass(frozen=True)
class P:
    """One scalar parameter plus the metadata that decides what may be done to it."""
    lo: float | None = None
    hi: float | None = None
    Step: float | None = None
    v: float | None = None
    name: str = None          # Name of the variable
    tf: str = "linear"          # "linear" or "log" interpolation inside [lo, hi]
    note: str = ""
    comment: str = ""
    
 
    def __post_init__(self):
        if self.lo is None or self.hi is None:
            raise ValueError(f"a fitted parameter needs bounds: {self}")
        if self.tf == "log" and self.lo <= 0:
            raise ValueError(f"log transform needs lo > 0: {self}")
 
    def return_Slider(self):
        """Return a Floatslider that can be used to adjust this parameter 
        in an interactive plot"""
    
        return W.FloatSlider(min=self.lo, max=self.hi, step=self.Step, value=self.v,
                                           description=self.note, continuous_update=False)
        
    
def _as_p(default, value, where):
    """A bare number -> the declared `P` with only its value replaced.

    The RANGE belongs to the class and is a hard boundary, so anything outside it raises here
    """
    if isinstance(value, P):
        return value
    v = float(value)
    if not (default.lo <= v <= default.hi):
        raise ValueError(f"{where} = {v} is outside its declared range "
                         f"[{default.lo}, {default.hi}]  ({default.comment or default.name})")
    return dataclasses.replace(default, v=v)

def _log_ok(p):
    return p.tf == "log" and p.lo > 0 and p.hi > 0


def to_u(p, v=None):
    """value -> u in [0, 1]."""
    v = p.v if v is None else v
    if _log_ok(p):
        lo, hi, v = np.log(p.lo), np.log(p.hi), np.log(max(float(v), 1e-300))
    else:
        lo, hi, v = p.lo, p.hi, float(v)
    span = hi - lo
    return 0.5 if abs(span) < 1e-300 else float((v - lo) / span)


def from_u(p, u):
    """u in [0, 1] -> value. Clipped, because ParamHolder treats the bounds as hard."""
    u = float(np.clip(u, 0.0, 1.0))
    if _log_ok(p):
        return float(np.exp(np.log(p.lo) + u * (np.log(p.hi) - np.log(p.lo))))
    return float(p.lo + u * (p.hi - p.lo))

def narrow(p, lo, hi, v=None):
    """A copy of p with tighter bounds, and its value pulled inside them.
    """
    lo, hi = float(lo), float(hi)
    v = p.v if v is None else v
    return dataclasses.replace(p, lo=lo, hi=hi, v=float(np.clip(v, lo, hi)))


def make_nuclear(comp, mu=(-0.85, -0.15), width=(0.25, 0.85)):
    """Force a noise component to live inside the nucleus, by bounds.
    """
    comp.mu = narrow(comp.mu, mu[0], min(mu[1], NUCLEAR_TAU_MAX), -0.55)
    comp.width = narrow(comp.width, width[0], width[1], 0.55)
    return comp


def dapi_marker(components=None, amp=(1.4, 2.6), strength=(0.6, 1.6), name=DAPI_NAME):
    comps = components if components is not None else [
        ClusterNoise(w=.8, s=1.2, mu=-.55, width=.55, sharp=4., scale=.30, clust=1.2,
                     fill=.45, soft=.30),
        BlobNoise(w=.5, s=1.1, mu=-.60, width=.65, sharp=3., scale=.40),
    ]
    for c in comps:
        make_nuclear(c)
        c.s = narrow(c.s, strength[0], strength[1], float(np.clip(c.s.v, *strength)))
    m = PanelMarker(name=name, fluorophore=DAPI_DYE, noise_components=comps,
                    polarity=P(-2, 2, .1, 0.0, "polarity"))
    m.amp = narrow(m.amp, amp[0], amp[1], float(np.clip(m.amp.v, *amp)))
    return m


class ParamHolder:
    """Mixin: any field declared with a `P` default may be set with a plain number.

        ClusterNoise(w=.80, fill=.22)      instead of
        ClusterNoise(w=P(0, 1, .05, .80, "weight"), fill=P(.02, 1, .02, .22, "fill frac"))

    The bounds, step, name and comment all come from the class declaration, so an instance only
    ever says WHAT THIS ONE IS -- and cannot silently widen a range by restating it wrongly.
    Passing a `P` explicitly still works if you genuinely need different bounds.
    """

    def __post_init__(self):
        for f in dataclasses.fields(self):
            default, cur = f.default, getattr(self, f.name)
            where = f"{type(self).__name__}.{f.name}"
            if isinstance(default, P):
                if not isinstance(cur, P):
                    setattr(self, f.name, _as_p(default, cur, where))
            elif (isinstance(default, tuple) and default
                  and all(isinstance(x, P) for x in default)):
                setattr(self, f.name, tuple(
                    _as_p(d, c, f"{where}[{i}]")
                    for i, (d, c) in enumerate(zip(default, cur))))
        
# 2. Component Dataclasses
@dataclass
class CellGeometry(ParamHolder):
    RADIUS: P = P(
        4.0,
        48.0,
        0.5,
        24.0,
        "radius",
        note="radius",
        comment="equivalent-sphere radius; volume is normalised to (4/3)pi R^3 whatever the roughness.",
    )
    ROUGH: P = P(
        0.0,
        0.50,
        0.01,
        0.25,
        "rough",
        note="rough",
        comment="SD of log-radius: 0.25 ~ +-25% radial wobble. Orthogonal to radius and beta.",
    )
    BETA: P = P(
        0.5,
        4.0,
        0.1,
        1.9,
        "beta",
        note="beta",
        comment="spectral tilt at FIXED total amplitude. Large -> a few fat lobes; "
        "small -> finer crenulation. Does NOT change how rough the cell is.",
    )
    ELONG: P = P(
        0.3,
        3.0,
        0.05,
        1.6,
        "elong",
        note="elong",
        comment="volume-preserving aspect ratio. >1 prolate (rod), 1 = sphere, <1 oblate.",
    )

    # 2) NUCLEUS
    NUC_FRAC: P = P(
        0.10,
        0.80,
        0.01,
        0.25,
        "nuc_frac",
        note="nuc frac",
        comment="nucleus:cell equivalent-RADIUS ratio -> volume ratio is nuc_frac**3.",
    )
    NUC_ROUGH: P = P(
        0.0,
        0.50,
        0.01,
        0.50,
        "nuc_rough",
        note="nuc rough",
        comment="nuclear log-radius SD. Orthogonal to radius and beta.",
    )
    NUC_BETA: P = P(
        0.5,
        6.0,
        0.1,
        3.2,
        "nuc_beta",
        note="nuc beta",
        comment="nuclear spectral tilt. Large -> a few fat lobes; small -> finer crenulation.",
    )
    NUC_CORR: P = P(
        0.0,
        1.0,
        0.05,
        0.40,
        "nuc_corr",
        note="nuc corr",
        comment="how much the nuclear outline mirrors the cell's",
    )
    NUC_OFFSET: P = P(
        0.0,
        1.0,
        0.05,
        1.0,
        "nuc_offset",
        note="nuc offset",
        comment="nuclear displacement as a fraction of the free cytoplasmic room. "
        "0 puts the nucleus exactly on the tessellation seed -> trivially recoverable.",
    )
    RIM: P = P(
        0.0,
        8.0,
        0.5,
        0.0,
        "rim",
        note="rim",
        comment="minimum cytoplasm between nuclear and plasma membrane, in lateral voxels.",
    )

    # 3) ORIENTATION   (ZYZ: polar+azim aim the long axis, roll spins about it)
    POLAR_DEG: P = P(0.0, 180.0, 5.0, 65.0, "polar_deg", note="polar")
    AZIM_DEG: P = P(0.0, 360.0, 5.0, 25.0, "azim_deg", note="azim")
    ROLL_DEG: P = P(0.0, 360.0, 5.0, 0.0, "roll_deg", note="roll")
    
    


@dataclass
class BaseNoise(ParamHolder):
    """Base class containing the parameters shared by ALL noise types."""
    scope: str = "base" # Will be overwritten by subclasses
    w: P = P(0, 1, .05, .50, "weight")
    s: P = P(0, 2, .05, 1.00, "strength")
    mu: P = P(-1, 1, .05, -0.50, "mu")
    width: P = P(.05, 1.5, .05, .60, "width")
    sharp: P = P(.5, 10, .5, 4.0, "sharp")
    
    def field_fct(self, ctx):
        raise NotImplementedError(f"{type(self).__name__} defines no field_fct")

@dataclass
class FlatNoise(BaseNoise):
    """No spatial texture at all, used for the fussel that binds uniformly
    """

    scope: str = "flat"

    def field_fct(self, ctx):
        return np.ones(ctx.noise.shape, np.float32)

@dataclass
class BlobNoise(BaseNoise):
    scope: str = "blob"
    scale: P = P(.1, 3, .05, 0.40, "grain um")
    
    def field_fct(self, ctx):
        H = np.exp(-2 * np.pi ** 2 * ctx.px(self.scale) ** 2 * ctx.f2)
        return np.exp(self.s.v * _filtered(ctx.noise, H))

@dataclass
class ClusterNoise(BaseNoise):
    scope: str = "cluster"
    scale: P = P(.1, 1.5, .05, 0.30, "speckle um")
    clust: P = P(.5, 8, .1, 1.40, "cluster um")
    fill: P = P(.02, 1, .02, 0.35, "fill frac")
    soft: P = P(.05, 1, .05, 0.30, "gate soft")    
    
    def field_fct(self, ctx):
        fine = _filtered(ctx.noise, np.exp(-2 * np.pi ** 2 * ctx.px(self.scale) ** 2 * ctx.f2))
        coarse = _filtered(ctx.gate, np.exp(-2 * np.pi ** 2 * ctx.px(self.clust) ** 2 * ctx.f2))
        fill = float(np.clip(self.fill.v, 1e-3, 1 - 1e-3))
        soft = max(float(self.soft.v), 1e-3)
        # ndtri(1-fill) is the exact Gaussian quantile, so `fill` IS the occupied fraction
        gate = 1.0 / (1.0 + np.exp(-(coarse - ndtri(1.0 - fill)) / soft))
        return (gate * np.exp(self.s.v * fine)).astype(np.float32)
    
@dataclass
class NetworkNoise(BaseNoise):
    scope: str = "network"
    scale: P = P(.2, 3, .05, 0.70, "mesh um")
    coherence: P = P(.05, 1, .05, 0.30, "coherence")

    def field_fct(self, ctx):
        f0 = 1.0 / ctx.px(self.scale)
        sf = f0 / max(float(self.coherence.v), 1e-3)
        H = np.exp(-(np.sqrt(ctx.f2) - f0) ** 2 / (2 * sf ** 2))
        return np.exp(self.s.v * _filtered(ctx.noise, H))

@dataclass
class FibreNoise(BaseNoise):
    scope: str = "fibre"
    sharp: P = P(.5, 10, .5, 4.0, "sharp")
    lam: P = P(.1, 1.5, .05, 0.25, "thickness um")
    length: P = P(1, 20, .5, 6.0, "length um")

    def field_fct(self, ctx):
        H = np.exp(-2 * np.pi ** 2 * (ctx.px(self.lam) ** 2 * ctx.fperp2
                                      + ctx.px(self.length) ** 2 * ctx.fpar2))
        return np.exp(self.s.v * _filtered(ctx.noise, H))

@dataclass
class SheetNoise(BaseNoise):
    scope: str = "sheet"
    lam: P = P(.5, 5, .1, 1.90, "band period um")
    coherence: P = P(.05, 1, .05, 0.35, "coherence")
    length: P = P(1., 20., .5, 6.0, "across um")
    
    def field_fct(self, ctx):
        f0 = 1.0 / ctx.px(self.lam)
        sf = f0 / max(float(self.coherence.v), 1e-3)
        H = (np.exp(-(np.sqrt(ctx.fpar2) - f0) ** 2 / (2 * sf ** 2))
             * np.exp(-2 * np.pi ** 2 * ctx.px(self.length) ** 2 * ctx.fperp2))
        return np.exp(self.s.v * _filtered(ctx.noise, H))

@dataclass
class PanelMarker(ParamHolder):
    name: str = "Unnamed Marker"
    fluorophore: str = "FITC"
    amp: P = P(.1, 3, .1, 2.0, "amp",
               comment="mean intensity inside a cell expressing this marker at level 1. "
                       "amp x level x E_PER_UNIT = photons per cell.")
    polarity: P = P(-2, 2, .1, 0.0, "polarity",
                    comment="von Mises-Fisher lobe strength along pol_dir. 0 = isotropic.")
    pol_dir: Tuple[P, P, P] = (P(-np.pi, np.pi, .1, 0.0, "pol dir x"),
                               P(-np.pi, np.pi, .1, 1.0, "pol dir y"),
                               P(-np.pi, np.pi, .1, 2.0, "pol dir z"))
    # A list to hold any combination of noise components
    noise_components: List[BaseNoise] = field(default_factory=list)
    
    artifact_affinity: P = P(
        0.0,
        2.0,
        0.05,
        0.6,
        "artifact affinity",
        note="art aff",
        comment="how strongly this marker's antibody binds artifacts",
    )
     
    def __post_init__(self):
        super().__post_init__()
        for noise in self.noise_components:
            assert isinstance(noise, BaseNoise), f"This noise: {type(noise)} is not defined!"
            

@dataclass
class MarkerPanel:
    """Every marker imaged in one experiment. Shared by all cell types in the image."""
    Markers: Dict[str, PanelMarker] = field(default_factory=dict)
    
    def __post_init__(self):
        # an empty panel is "not configured yet", not an error
        if not self.Markers:
            return                      
        d = self.Markers.get(DAPI_NAME)
        if d is None:
            raise ValueError(
                f"every panel needs a {DAPI_NAME!r} marker.\
                    Build one with parameter.dapi_marker().")
        if d.fluorophore != DAPI_DYE:
            raise ValueError(f"{DAPI_NAME} must use the {DAPI_DYE!r} dye, got "
                             f"{d.fluorophore!r}")
        clash = [n for n, m in self.Markers.items()
                 if n != DAPI_NAME and m.fluorophore == DAPI_DYE]
        if clash:
            raise ValueError(f"{DAPI_DYE!r} is reserved for {DAPI_NAME}; also used by {clash}")
        bad = [c.scope for c in d.noise_components if c.mu.hi > NUCLEAR_TAU_MAX]
        if bad:
            raise ValueError(
                f"{DAPI_NAME} must be nuclear: every component needs mu.hi <= "
                f"{NUCLEAR_TAU_MAX} (tau is -1 at the nucleus centre, 0 at the envelope, "
                f"+1 at the membrane). Offending components: {bad}. Use dapi_marker(), which "
                f"narrows the bounds so no prior draw or fit can leave the nucleus.")
    
    @property
    def dapi(self):
        return self.Markers.get(DAPI_NAME)

    @property
    def names(self):
        return list(self.Markers)
    
    @property
    def fluorophores(self):
        """marker name -> dye, the mapping the optics and the detector need."""
        return {k: m.fluorophore for k, m in self.Markers.items()}

    def n_pools(self):
        """Pools the tape must hold: ONE PER (marker, component) pair, not per component.

        Each marker gets its own consecutive block, so two markers using the same noise kind
        still draw independent frozen fields. 
        """
        return sum(len(m.noise_components) for m in self.Markers.values())
    
@dataclass
class CellType(ParamHolder):
    """A cell type is its shape plus HOW MUCH of each panel marker it expresses.

    `Expression` maps a marker name to a level: 0 is negative, 1 is the panel's nominal
    brightness, above 1 is bright. A level multiplies the rendered volume, which is exactly
    equivalent to scaling `PanelMarker.amp` but cannot run out of `amp`'s declared bounds.
    Markers missing from `Expression` are treated as not expressed.
    """
    name: str = "Unnamed"
    Color: str = "black"
    Geometry: CellGeometry = field(default_factory=CellGeometry)
    Expression: Dict[str, P] = field(default_factory=dict)

    def level(self, marker_name):
        p = self.Expression.get(marker_name)
        return 0.0 if p is None else float(p.v if hasattr(p, "v") else p)

    def expressed(self, thresh=1e-3):
        return [k for k in self.Expression if self.level(k) > thresh]
    

@dataclass
class ArtifactMap(ParamHolder):
    """Artifact map. Artifacts only appear where there is tissue for them to stick to."""

    MIN_DIST: P = P(
        2.0,
        200.0,
        1.0,
        12.0,
        "art min dist",
        note="min dist",
        comment="hard-core spacing between artifacts, voxels. Sets a ceiling on how "
        "many can be placed.",
    )
    SUPPORT_SCALE: P = P(
        2.0,
        200.0,
        1.0,
        30.0,
        "art support scale",
        note="support",
        comment="correlation length of the artifact-support field.",
    )
    SUPPORT_SCALE_Z: P = P(
        1.0,
        200.0,
        1.0,
        25.0,
        "art support scale z",
        note="support z",
        comment="axial correlation length (Barely relevant).",
    )
    COVER: P = P(
        0.05,
        1.0,
        0.05,
        0.60,
        "art cover",
        note="cover",
        comment="fraction of the field where artifacts are allowed, as a quantile. "
        "1.0 = anywhere there is tissue.",
    )


@dataclass
class Fussel(ParamHolder):
    """A lint fibre or eyelash lying on the plate, coated in non-specifically bound antibody.
    """

    N: P = P(
        0.0,
        4.0,
        1.0,
        1.0,
        "n fussel",
        note="n fibres",
        comment="how many fibres to place.",
    )
    WIDTH_UM: P = P(
        0.4,
        12.0,
        0.1,
        2.0,
        "fussel width",
        tf="log",
        note="width um",
        comment="fibre diameter.",
    )
    WIDTH_SIGMA: P = P(
        0.0,
        0.6,
        0.05,
        0.25,
        "fussel width sd",
        note="width sd",
        comment="lognormal spread of width between fibres:",
    )
    P_END_INSIDE: P = P(
        0.0,
        1.0,
        0.05,
        0.95,
        "fussel end inside",
        note="end inside",
        comment="Per end probability, that the fibre terminates inside the frame instead "
        "of running off the edge.",
    )
    END_TRIM: P = P(
        0.05,
        0.45,
        0.05,
        0.30,
        "fussel end trim",
        note="end trim",
        comment="how far in an inside-terminating end may sit, as a fraction of the "
        "chord. Capped below 0.5 so the two ends can never cross and leave "
        "nothing to draw.",
    )
    WOBBLE_UM: P = P(
        0.0,
        20.0,
        0.5,
        6.0,
        "fussel wobble",
        note="wobble um",
        comment="RMS lateral deviation from the straight entry-to-exit chord. 6 um on a "
        "52 um frame is a clear worm; 0 is a straight.",
    )
    BETA: P = P(
        0.5,
        4.0,
        0.1,
        2.0,
        "fussel beta",
        note="beta",
        comment="spectral tilt of the wobble at FIXED RMS -- high = one long smooth bend, "
        "low = kinked.",
    )
    Z_FRAC: P = P(
        0.0,
        1.0,
        0.05,
        0.5,
        "fussel z",
        note="z frac",
        comment="axial centre toward the coverslip, as a fraction of the half-section. "
        "1.0 is the coverslip itself.",
    )
    Z_WOBBLE_FRAC: P = P(
        0.0,
        1.0,
        0.05,
        0.90,
        "fussel z wobble",
        note="z wobble",
        comment="axial meander along the fibre, as a fraction of the room below it.",
    )
    MU: P = P(
        0.0,
        0.3,
        0.05,
        0.0,
        "fussel mu",
        note="mu",
        comment="Centre of the expression band in the blob's tau",
    )
    WIDTH: P = P(
        1.25,
        1.5,
        0.05,
        1.40,
        "fussel band",
        note="band",
        comment="width.",
    )
    SHARP: P = P(
        0.5,
        10.0,
        0.5,
        8.0,
        "fussel band sharp",
        note="band sharp",
        comment="Sharpness of the cutoff.",
    )
    GAIN: P = P(
        0.5,
        500.0,
        0.5,
        60.0,
        "fussel gain",
        tf="log",
        note="gain",
        comment="brightness as a multiple of the contaminated marker's own amp.",
    )
    GAIN_SIGMA: P = P(
        0.0,
        1.0,
        0.05,
        0.30,
        "fussel gain sd",
        note="gain sd",
        comment="lognormal brightness spread between fibres.",
    )


@dataclass
class Aggregate(ParamHolder):
    """Precipitated conjugated antibody: a small, bright, roughly circular blob.
    """

    N: P = P(
        0.0,
        32.0,
        1.0,
        6.0,
        "n aggregates",
        note="n aggs",
        comment="how many to place in the frame.",
    )
    DIAM_UM: P = P(
        0.3,
        4.0,
        0.1,
        1.0,
        "agg diameter",
        tf="log",
        note="diam um",
        comment="median Aggregate diameter.",
    )
    DIAM_SIGMA: P = P(
        0.0,
        1.0,
        0.05,
        0.45,
        "agg diameter sd",
        note="diam sd",
        comment="lognormal size spread.",
    )
    ROUGH: P = P(
        0.0,
        0.30,
        0.01,
        0.12,
        "agg rough",
        note="rough",
        comment="SD of log-radius: 0.25 ~ +-25% radial wobble. Orthogonal to radius and beta.",
    )
    MU: P = P(
        -1.0,
        1.0,
        0.05,
        0.0,
        "agg mu",
        note="mu",
        comment="centre of the expression band in the blob's tau.",
    )
    WIDTH: P = P(
        0.05,
        1.5,
        0.05,
        1.20,
        "agg band",
        note="band",
        comment="width of that band. Wide, so the blob is solid rather than a shell.",
    )
    SHARP: P = P(
        0.5,
        10.0,
        0.5,
        6.0,
        "agg band sharp",
        note="band sharp",
        comment="Sharpness of the cutoff.",
    )
    GAIN: P = P(
        1.0,
        5000.0,
        1.0,
        150.0,
        "agg gain",
        tf="log",
        note="gain",
        comment="Brightness as a multiple of the marker's amp.",
    )
    GAIN_SIGMA: P = P(
        0.0,
        1.5,
        0.05,
        0.70,
        "agg gain sd",
        note="gain sd",
        comment="lognormal brightness spread.",
    )
    Z_FRAC: P = P(
        -1.0,
        1.0,
        0.05,
        0.60,
        "agg z",
        note="z frac",
        comment="signed band centre as a fraction of the half-section.",
    )
    Z_SIGMA_FRAC: P = P(
        0.0,
        1.0,
        0.05,
        0.40,
        "agg z sd",
        note="z sd",
        comment="axial scatter between aggregates.",
    )
    SPILL: P = P(
        0.0,
        0.3,
        0.01,
        0.0,
        "agg spill",
        note="spill",
        comment="fraction leaking into the OTHER antibody channels (bleed-through, or a "
        "clump carrying two conjugates).",
    )
    
@dataclass
class ArtifactMask(ParamHolder):
    """How to build the 2D artifact mask from  the 3D labels."""

    MASK_PCT: P = P(
        0.50,
        0.999,
        0.005,
        0.95,
        "art mask pct",
        note="mask pct",
        comment="coverage quantile for the artifact's own 2D footprint.",
    )
    DILATE_PX: P = P(
        0.0,
        10.0,
        1.0,
        2.0,
        "art dilate",
        note="dilate px",
        comment="grow the reported extent by this much.",
    )

@dataclass
class Artifacts(ParamHolder):
    """Everything about artifacts, hung off the world as `AR` so config.PIN's `^AR\\.` reaches
    all of it."""

    Map: ArtifactMap = field(default_factory=ArtifactMap)
    Fussel: Fussel = field(default_factory=Fussel)
    Aggregate: Aggregate = field(default_factory=Aggregate)
    Mask: ArtifactMask = field(default_factory=ArtifactMask)
    
@dataclass
class Detector(ParamHolder):
    """Everything that happens AFTER the optics. Not a property of any cell."""
    AF_SCALE_UM: P = P(1., 100., 1., 25.0, "af scale", note="af scale",
                       comment="spatial scale of the autofluorescence field, in microns")
    AF_CV: P = P(0., 1.5, .05, 0.45, "af cv", note="af cv",
                 comment="relative SD of the autofluorescence field. 0 = perfectly flat")
    ILLUM_CV: P = P(0., .5, .01, 0.06, "illum cv", note="illum cv",
                    comment="random flat-field non-uniformity, as a fraction")
    VIGNETTE: P = P(0., .6, .05, 0.18, "vignette", note="vignette",
                    comment="radial illumination falloff at the frame corners, as a fraction")
    E_PER_UNIT: P = P(1., 2000., 10., 120.0, "e per unit", note="e/unit",
                      comment="photoelectrons per unit of marker amplitude. THIS sets the "
                              "shot-noise level: doubling it halves the relative noise.")
    READ_E: P = P(0., 50., .5, 2.5, "read noise", note="read e-",
                  comment="camera read noise, electrons RMS. Dominates where the signal is dark.")
    DARK_E: P = P(0., 200., 1., 5.0, "dark", note="dark e-",
                  comment="dark current + stray light, in electrons")
    ADU_PER_E: P = P(.05, 10., .05, 0.5, "adu per e", note="adu/e-",
                     comment="digitiser conversion gain")
    OFFSET_ADU: P = P(0., 2000., 10., 100.0, "offset", note="offset",
                      comment="camera black level")
    BIT_DEPTH: int = 16


# Optics: 
@dataclass(frozen=True)
class Optics:
    """Known Physical properties of the detector (Macsima) and the experiment."""
    um_per_px: float = 0.325 
    um_per_pz: float = 0.325 
    
    focal_um: float = 0.0
    
    # VERIFY
    # Numerical Aperture (NA)
    na: float = 0.45 #or 0.75
     
    wavelength_um: float = 0.530 # fallback
    
    # different refractive index of tissue and medium. VERIFY
    n_immersion: float = 1.0
    n_sample: float = 1.33
    
    # Thickness of the section
    section_um: float = 4.
    # Depth of the section CENTRE below the coverslip
    depth_um: float = 2.0
    
    @property
    def sample_depth_um(self):
        """Depth of the section centre below the coverslip."""
        return max(self.depth_um, self.section_um / 2)
    
    @property
    def tan_theta(self):
        return float(np.tan(np.arcsin(np.clip(self.na / self.n_immersion, 0.0, 0.999))))

@dataclass
class Tissue:    
    Panel: MarkerPanel = field(default_factory=MarkerPanel)
    CellTypes: Dict[str, CellType] = field(default_factory=dict)
    # Tissue params...
    

@dataclass
class TissueGeometry(ParamHolder):
    """How cells are laid out in the volume. Lengths in VOXELS, like CellGeometry."""
    MIN_DIST: P = P(2., 60., .5, 11.0, "min dist", note="min dist",
                    comment="hard-core spacing: no two centres closer than this. Roughly "
                            "1.2-1.8 x radius; too large and few candidates survive.")
    SUPPORT_SCALE: P = P(2., 200., 1., 40.0, "support scale", note="support",
                    comment="correlation length of the tissue-support field. Keep >> cell size "
                            "or the support boundary starts looking like a cell edge.")
    SUPPORT_SCALE_Z: P = P(1., 200., 1., 25.0, "support scale z", note="support z",
                    comment="AXIAL correlation length, separately.")
    COVER: P = P(.05, 1., .05, .75, "cover", note="cover",
                    comment="fraction of the volume that is tissue. Applied as a quantile of "
                            "the support field, so it maps monotonically onto realised cover.")
    GROW: P = P(1., 3., .05, 1.35, "grow", note="grow",
                    comment="how far a cell may claim, in units of its own boundary. 1 = free "
                            "shapes with gaps between them; ~2 = confluent.")
    SIZE_SIGMA: P = P(0., .6, .01, .15, "size sigma", note="size sd",
                    comment="lognormal spread of cell size: r = RADIUS * exp(sigma * z).")
    NUC_FRAC_SIGMA: P = P(0., .6, .01, .15, "nuc frac sigma", note="nucfrac sd",
                    comment="spread of the nucleus:cell ratio. Keep > 0, or nuclear size "
                            "predicts cell size exactly and the segmentation is invertible.")
    NEIGH_RADIUS: P = P(2., 100., 1., 24.0, "neigh radius", note="neigh r",
                    comment="radius of the neighbour graph, voxels -- what 'nearby' means for "
                            "neighbourhood-dependent marker expression.")

    
@dataclass
class CellContext:
    """What a rule may look at when deciding one cell's expression."""
    label: int
    # candidate index in the tape
    index: int
    # voxel coords         
    centre: np.ndarray
    geom: CellGeometry
    # labels within NEIGH_RADIUS
    neigh_labels: list
    # label -> type name, for cells already assigned
    types: dict

    def n_neighbours(self):
        return len(self.neigh_labels)

    def neighbour_types(self):
        return [self.types[j] for j in self.neigh_labels if j in self.types]
    
    
    
def get_all_parameters(obj, prefix=""):
    """Recursively fetches all P instances from dataclasses, dicts, and lists."""
    params = {}    
    
    if isinstance(obj, P):
        return {prefix: obj} if prefix else {}
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            params.update(get_all_parameters(value, new_prefix))
            
    elif isinstance(obj, (list, tuple)):
        # Handle lists by appending the index to the prefix (e.g., noise_components[0])
        for i, item in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            params.update(get_all_parameters(item, new_prefix))
            
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            attr = getattr(obj, f.name)
            new_prefix = f"{prefix}.{f.name}" if prefix else f.name
            
            if isinstance(attr, P):
                params[new_prefix] = attr
            else:
                params.update(get_all_parameters(attr, new_prefix))
                
    return params


DEFAULT_PIN = (r"noise_components\[0\]\.w$",) + (r"^DET\.",)
_IDX = re.compile(r"^(.+?)\[(\d+)\]$")

def _step(obj, name, idx):
    obj = obj[name] if isinstance(obj, dict) else getattr(obj, name)
    return obj if idx is None else obj[idx]


def _split(part):
    m = _IDX.match(part)
    return (m.group(1), int(m.group(2))) if m else (part, None)


def _resolve(root, path):
    """Walk to the parent of `path` and return (parent, attr_name, index_or_None)."""
    parts = path.split(".")
    cur = root
    for part in parts[:-1]:
        name, idx = _split(part)
        cur = _step(cur, name, idx)
    name, idx = _split(parts[-1])
    return cur, name, idx


def get_p(root, path):
    """The `P` object living at `path`."""
    parent, name, idx = _resolve(root, path)
    return _step(parent, name, idx)


def set_p(root, path, newp):
    """Replace the `P` at `path` in place. Handles tuple fields (pol_dir) by rebuilding them."""
    parent, name, idx = _resolve(root, path)
    if idx is None:
        if isinstance(parent, dict):
            parent[name] = newp
        else:
            setattr(parent, name, newp)
        return
    seq = parent[name] if isinstance(parent, dict) else getattr(parent, name)
    if isinstance(seq, tuple):
        lst = list(seq)
        lst[idx] = newp
        seq = tuple(lst)
        if isinstance(parent, dict):
            parent[name] = seq
        else:
            setattr(parent, name, seq)
    else:
        seq[idx] = newp


def fitted_paths(root, pin=DEFAULT_PIN, keep=None):
    """Sorted, deterministic list of the paths that go into theta.
    """
    paths = sorted(get_all_parameters(root))
    if pin:
        paths = [p for p in paths if not any(re.search(pat, p) for pat in pin)]
    if keep:
        paths = [p for p in paths if any(re.search(pat, p) for pat in keep)]
    return paths


def sample_prior(rng, paths):
    """Uniform in NORMALISED space, i.e. uniform in value, or log-uniform where tf='log'."""
    return rng.random(len(paths))

def bounds(root, paths):
    """(lo, hi) per path in raw units -- for reporting predictions in physical terms."""
    ps = [get_p(root, p) for p in paths]
    return (np.array([p.lo for p in ps], float), np.array([p.hi for p in ps], float))


def set_vector(root, paths, u):
    """Write u back into the tree, in place. Returns `root` for chaining."""
    u = np.asarray(u, float).ravel()
    if u.size != len(paths):
        raise ValueError(f"theta has {u.size} entries but {len(paths)} paths were given")
    for path, ui in zip(paths, u):
        p = get_p(root, path)
        set_p(root, path, dataclasses.replace(p, v=from_u(p, ui)))
    return root

NOISE_KINDS = (BlobNoise, ClusterNoise, NetworkNoise, FibreNoise, SheetNoise)

DEVICE_DETECTOR = Detector(
    E_PER_UNIT=120.0,
    READ_E=2.5,
    DARK_E=5.0,
    ADU_PER_E=0.5,
    OFFSET_ADU=100.0,
    AF_SCALE_UM=25.0,
    AF_CV=0.45,
    ILLUM_CV=0.06,
    VIGNETTE=0.18,
)