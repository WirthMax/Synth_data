# grid and physical scale
# microns per LATERAL voxel -- the conversion, not a resolution knob
UM_PER_VOX = 0.325
# axial voxel this many times COARSER than lateral
Z_RATIO = 4.0
# (sz, sy, sx) voxel size per axis, in lateral units
SPACING = (Z_RATIO, 1.0, 1.0)

CELL_VOL = (128, 128, 128)  # single-cell sandbox
TISSUE_VOL = (6, 160, 160)  # tissue block: 13 x 52 x 52 um at 0.325 um/vox

# shape basis
# lowest SH degree: 0 is pure size (absorbed by the volume normalisation), 1 shifts the centroid
# off the seed, 2 is the lowest true shape mode. 4 is the ceiling of the hardcoded Cartesian forms.
L_MIN = 2
L = 4

# the tape
# None = a fresh tissue every run; set an int to reproduce one
SEED = None
# candidate cells drawn once; thinning selects from these
N_CAND = 1500
# 2D sandbox size
SIZE = 201
# 2D tissue tile
TILE = 256
# Fourier modes for the 2D boundary
K = 20

# frozen noise fields
N_POOLS = 12
# Beyond the bank, fields are reused at a deterministic shift
POOL_BANK = 16

# optics (the instrument)
OPTICS = dict(
    um_per_px=UM_PER_VOX,
    um_per_pz=UM_PER_VOX * Z_RATIO,
    # focal plane relative to the section CENTRE
    focal_um=0.0,
    # VERIFY against the objective in use (or 0.75)
    na=0.45,
    # fallback only; per-marker wavelength comes from FLUOROPHORES
    wavelength_um=0.530,
    # air objective
    n_immersion=1.0,
    # tissue / mounting medium
    n_sample=1.33,
    # thickness of the physical section
    section_um=4.0,
    # depth of the section CENTRE below the coverslip
    depth_um=2.0,
)

# Emission wavelength per dye in microns.
FLUOROPHORES = {"DAPI": 0.461, "FITC": 0.519, "PE": 0.578, "APC": 0.660}
# Tissue autofluorescence per dye
AUTOFLUOR = {"DAPI": 0.20, "FITC": 0.60, "PE": 0.30, "APC": 0.10}

# dyes an ANTIBODY marker may draw -- DAPI's is reserved for the Nuclear stain
DYES = ("FITC", "PE", "APC")
DAPI_NAME = "DAPI"
DAPI_DYE = "DAPI"
NUCLEAR_TAU_MAX = 0.0

# Normalised distance is clipped to this OUTSIDE the object.
_TAU_CLIP = 4.0

# Camera
DETECTOR = dict(
    # photoelectrons per unit of marker amplitude
    E_PER_UNIT=120.0,
    # camera read noise, electrons RMS
    READ_E=2.5,
    # dark current + stray light, electrons
    DARK_E=5.0,
    # digitiser conversion gain
    ADU_PER_E=0.5,
    # black level
    OFFSET_ADU=100.0,
    # autofluorescence correlation length, microns
    AF_SCALE_UM=25.0,
    # relative SD of the autofluorescence field
    AF_CV=0.45,
    # random flat-field non-uniformity
    ILLUM_CV=0.06,
    # radial falloff at the frame corners
    VIGNETTE=0.18,
)
BIT_DEPTH = 16
ADU_MAX = 2**BIT_DEPTH - 1

PIN_DEGENERATE = (r"noise_components\[0\]\.w$",)
PIN_GROUNDED = (r"^DET\.",)
PIN_ARTIFACT = (r"^AR\.",)
PIN = PIN_DEGENERATE + PIN_GROUNDED + PIN_ARTIFACT
KEEP = [r"\.scale$", r"\.clust$", r"\.fill$", r"\.s$", r"\.amp$", r"^AR\.", r"^ARCH\."]


ARTIFACTS = dict(n_cand=256, n_mode=16)
# cycle misregistration (AR.Cycle). Markers are imaged in cycles of 1-3; the reference group
# (DAPI, the registration target) is cycle 0 and never shifts. p_black_bars is the chance that the
# tile is a crop from the slide border, where a shifted cycle has no data and leaves a black band;
# otherwise the tile is an interior crop and the edge is filled (reflected).
CYCLE = dict(group_sizes=(1, 2, 3), ref_group_size=1, p_black_bars=0.5)

# ---- Phase 2: exact label export ----
# a cell is VISIBLE if its 2D cross-section covers at least min_area_px pixels; a type is PRESENT
# in a tile if it has at least min_visible_cells visible cells. Invisible cells and non-present
# types become the IGNORE label (255) in type_map, so no loss is charged where there is too little
# evidence to learn from.
VIS = dict(min_area_px=15, min_visible_cells=3)
# context = where a cell sits. Context 0 is always background tissue; the structure KINDS below get
# indices 1, 2, ... (fixed across samples so C is constant for the model). Add a kind here when a
# new BaseStructure subclass is introduced.
CONTEXT_KINDS = ("ordered", "vessel")
IGNORE_LABEL = 255
# This should be larger than the possible amount of cells that can be generated
ARTIFACT_ID0 = 100000
# marker localisation: a marker's intensity band sits at a continuous position `mu` along tau
# (-1 nucleus centre, 0 envelope, +1 plasma membrane). The model REGRESSES this. Antibody markers
# draw mu ~ U[MU_LO, MU_HI]; the nuclear counterstain is pinned at DAPI_MU. COMPARTMENT_MU are the
# three discrete band-centres kept only for a diagnostic "derived compartment" readout (thresholding
# mu), not for generation. Lives here (numpy-only module) so the synmodel training side can read it
# without importing the generation stack.
MU_LO, MU_HI = -0.9, 1.0
DAPI_MU = -0.6
COMPARTMENT_MU = {"nuclear": -0.50, "membrane": 1.00, "cytoplasmic": 0.40}

# ---- Phase 3: flexible cell types (plan 3.0, 3.7) ----
# type_mode selects the generator's biology sampler. "flexible" (the default, and the only path the
# v3 model trains on) redraws the whole panel, lineage-tree type set and per-context compositions
# per sample (flexible_types.build_flexible_world). "fixed" is the original per-dataset layout
# (panel/types/structures frozen once) -- kept only as a reference generation mode; the old flat-theta
# model it fed has been removed.
# CURRICULUM is the set of knobs that widen the flexible prior across training rounds (plan 3.7):
# the START column is the gentle early-round prior, FINAL the full prior the final fine-tune uses.
# A round picks values between the two; the final fine-tune (plan 6.4) must use FINAL only.
GENERATOR = dict(
    type_mode="flexible",
    curriculum=dict(
        #              knob                 start        final
        k_true_range=dict(start=(2, 8),   final=(1, 30)),   # number of distinct cell types
        max_depth=dict(start=1,           final=4),         # lineage tree depth (1 = roots only)
        min_level_diff=dict(start=0.6,    final=0.3),       # distinctness margin, decades
        m_range=dict(start=(20, 40),      final=(8, 40)),   # channels (markers) per tile
    ),
)


def curriculum_at(t):
    """The plan 3.7 curriculum interpolated to progress `t` in [0, 1]: 0 = start prior, 1 = final.
    Ranges interpolate endpoint-by-endpoint and round to int; max_depth steps linearly. Returns the
    dict build_flexible_world's `curriculum` argument expects (m_range, k_true_range, max_depth,
    min_level_diff)."""
    t = float(min(max(t, 0.0), 1.0))
    c = GENERATOR["curriculum"]

    def _lerp_range(key):
        s, f = c[key]["start"], c[key]["final"]
        return (int(round(s[0] + t * (f[0] - s[0]))), int(round(s[1] + t * (f[1] - s[1]))))

    return dict(
        k_true_range=_lerp_range("k_true_range"),
        m_range=_lerp_range("m_range"),
        max_depth=int(round(c["max_depth"]["start"]
                            + t * (c["max_depth"]["final"] - c["max_depth"]["start"]))),
        min_level_diff=float(c["min_level_diff"]["start"]
                             + t * (c["min_level_diff"]["final"] - c["min_level_diff"]["start"])),
    )


# ============================================================ amortized inversion
# sigma in [2.5e-3, 7.4]; theta lives in [0, 1]
LOGVAR_MIN, LOGVAR_MAX = -12.0, 4.0
