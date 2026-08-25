# grid and physical scale
UM_PER_VOX = 0.325          # microns per LATERAL voxel -- the conversion, not a resolution knob
Z_RATIO = 1.0               # axial voxel this many times COARSER than lateral
SPACING = (Z_RATIO, 1.0, 1.0)   # (sz, sy, sx) voxel size per axis, in lateral units

CELL_VOL = (128, 128, 128)  # single-cell sandbox
TISSUE_VOL = (40, 160, 160)  # tissue block: 13 x 52 x 52 um at 0.325 um/vox

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
ADU_MAX = 2 ** BIT_DEPTH - 1

PIN_DEGENERATE = (r"noise_components\[0\]\.w$",)
PIN_GROUNDED = (r"^DET\.",)
PIN = PIN_DEGENERATE + PIN_GROUNDED
KEEP = [r"\.scale$", r"\.clust$", r"\.fill$", r"\.s$", r"\.amp$"]


ARTIFACTS = dict(n_cand=256, n_mode=16)
# This should be larger than the possible amount of cells that can be generated
ARTIFACT_ID0 = 100000
# ============================================================ amortized inversion
LOGVAR_MIN, LOGVAR_MAX = -12.0, 4.0     # sigma in [2.5e-3, 7.4]; theta lives in [0, 1]
