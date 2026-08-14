import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from render import _filtered
import numpy as np
import ipywidgets as W
from scipy.special import ndtri 


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
    rather than quietly producing a nonsense texture several hundred lines later.
    """
    if isinstance(value, P):
        return value
    v = float(value)
    if not (default.lo <= v <= default.hi):
        raise ValueError(f"{where} = {v} is outside its declared range "
                         f"[{default.lo}, {default.hi}]  ({default.comment or default.name})")
    return dataclasses.replace(default, v=v)


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
    mu: P = P(-1, 1.5, .05, -0.50, "mu")
    width: P = P(.05, 1.5, .05, .60, "width")
    sharp: P = P(.5, 10, .5, 4.0, "sharp")
    
    def field_fct(self, ctx):
        raise NotImplementedError(f"{type(self).__name__} defines no field_fct")

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


NOISE_KINDS = (BlobNoise, ClusterNoise, NetworkNoise, FibreNoise, SheetNoise)

@dataclass
class CellMarker(ParamHolder):
    name: str = "Unnamed Marker"
    fluorophore: str = "FITC"
    amp: P = P(.1, 3, .1, 2.0, "amp",
               comment="mean intensity inside the cell. amp x E_PER_UNIT = photons per cell.")
    polarity: P = P(-2, 2, .1, 0.0, "polarity",
                    comment="von Mises-Fisher lobe strength along pol_dir. 0 = isotropic.")
    pol_dir: Tuple[P, P, P] = (P(-np.pi, np.pi, .1, 0.0, "pol dir x"),
                               P(-np.pi, np.pi, .1, 1.0, "pol dir y"),
                               P(-np.pi, np.pi, .1, 2.0, "pol dir z"))
    # A list to hold any combination of noise components
    noise_components: List[BaseNoise] = field(default_factory=list)
    
     
    def __post_init__(self):
        super().__post_init__()
        for noise in self.noise_components:
            assert type(noise) in NOISE_KINDS, f"This noise: {type(noise)} is not defined!"
            

@dataclass
class CellProfile:
    """Geometry plus every marker stained on this cell type."""
    Geometry: CellGeometry = field(default_factory=CellGeometry)
    Markers: Dict[str, CellMarker] = field(default_factory=dict)

    @property
    def fluorophores(self):
        """marker name -> dye, the mapping the optics and the detector need."""
        return {k: m.fluorophore for k, m in self.Markers.items()}

    def n_pools(self):
        """Largest component count over the markers -- the tape needs at least this many."""
        return max((len(m.noise_components) for m in self.Markers.values()), default=0)
    
    
    
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
    # Dictionary to hold multiple cell types, keyed by cell name
    CellTypes: Dict[str, CellProfile] = field(default_factory=dict)
    # Tissue params...

    
def get_all_parameters(obj, prefix=""):
    """Recursively fetches all P instances from dataclasses, dicts, and lists."""
    params = {}
    
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