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
N_CAND = 2500
# 2D sandbox size
SIZE = 201
# 2D tissue tile
TILE = 256
# Fourier modes for the 2D boundary
K = 20

# the reference tile edge every density statement in this file is made for (A = 160^2 px in
# flexible_types._expected_visible_cells, the strata edges, MIN_SPACING vs the query cap)
FIELD_REF = 160


def set_field(n):
    """Render a larger (or smaller) FIELD than the 160 px tile: an n x n px block, i.e. n x 0.325 um
    on a side (512 px = 166 um, 10x the area of a tile). The candidate pool grows with the AREA, so
    the seeds are exactly as dense as on a tile -- without it a 512 px field would have a tenth of
    the cells. Everything that depends on the field reads TISSUE_VOL / N_CAND at call time, so this
    must run before the first render, once per process (`run_v3.py gen --field` does it in every
    worker). n must be divisible by 32 for the model's strides. set_field(160) is the old tile,
    bit-identical."""
    global TISSUE_VOL, N_CAND
    n = int(n)
    if n % 32:
        raise ValueError(f"field {n} px is not divisible by 32 (the model's coarsest stride) -- "
                         f"use e.g. 160, 384, 512")
    TISSUE_VOL = (TISSUE_VOL[0], n, n)
    N_CAND = int(round(2500 * n * n / FIELD_REF ** 2))

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
    # camera read noise, electrons RMS (20 e- ~ 12 ADU single-frame, matching the real slide; it is
    # also what sets the backsub empty-tile floor -- see Detector.READ_E)
    READ_E=20.0,
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
    # --- MACSIMA background subtraction (staged-pipeline domain); see Detector for the physics ---
    # subtract an independent blank cycle from each non-DAPI marker and clip at 0 (1 = staged, 0 = raw)
    BG_SUBTRACT=1.0,
    # blank-vs-marker bleach/illumination drift (fraction); with BG_MARGIN_ADU sets the zero fraction
    BG_DRIFT=0.10,
    # blank-AF registration offset vs the marker (px); sets the residual's spatial texture
    BG_SHIFT_PX=1.0,
    # constant ADU over-subtracted after the blank; raises the empty-tile zero fraction
    BG_MARGIN_ADU=6.0,
    # fraction of defective/transient hot pixels (marker frame only -> survive backsub as the tail)
    HOT_RATE=7e-5,
    # electrons a hot pixel dumps
    HOT_E=500.0,
    # Gaussian sigma (px) correlating the read-noise floor; 0 = white (only active under BG_SUBTRACT)
    READ_SMOOTH_PX=0.5,
    # texture of the tissue background (PanelMarker.tissue_bg): correlation length um, relative SD
    TBG_TEX_UM=1.5,
    TBG_TEX_CV=0.5,
    # black level of the staged DAPI channel (offset-corrected, glass ~10 ADU on the real slide)
    DAPI_OFFSET_ADU=10.0,
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
# min_frac (2026-11-16): a cell must also show at least this share of its own nominal
# cross-section, pi x (short semi-axis)^2. The section is now cut from bulk, so every cell is also
# met as a thin cap above or below its body: a 20 px sliver of a 20 um cell is not a cell anyone
# would outline, and calling it one taught the head to fragment. Such slivers are IGNORE, not
# background -- they are still tissue and still lit.
VIS = dict(min_area_px=15, min_visible_cells=3, min_frac=0.08)
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

# Absolute brightness of antibody markers relative to the panel `amp` prior. The model sees only
# per-channel normalised images plus a RELATIVE strength, so absolute brightness is set by the
# training prior, not inferred -- but it still matters through shot-noise SNR and the zero fraction.
# Measured (auto_improve exp 000, DEV): at gain 1 the rendered marker channels had median 40-180 ADU
# and p99 0.7-4.8k against real backsub tissue tiles at median 1.4-4.8k and p99 6-40k (5-30x
# dimmer, 15% exact zeros vs 1% real). DAPI was only 2-3.6x dim, so the gain is applied to the
# MARKER expression range only (flexible_types.expression_range). 1.0 reproduces the old prior
# bit-for-bit. VERIFY: a later check put real marker medians at 100-700 against ~1.7k for gain-10
# renders, so 10 may overshoot.
# 2026-11-16: 30. A level is now the mean brightness INSIDE the marker's compartment
# (render.loc_norm), not the cell mean, so membrane and nuclear markers lost the 1 / volume-share
# boost they had (x2 to x600). 30 puts the expression range at 0.9-270 units = 54 ADU to 16k ADU in
# the stained compartment, against real per-channel p99.5 of 50 ADU (dead stains) to 60k (IgG,
# CD36, desmin). VERIFY against more slides.
EXPR_GAIN = 30.0

# Tissue background prior (PanelMarker.tissue_bg, drawn per channel per sample in flexible mode,
# log-uniform). The CALIBRATED centre, against 5 crops of the staged myocarditis slide, was marker
# (0.3, 25) / DAPI (1.5, 40): backsub marker channels in tissue median over channels 167 ADU (10-90%
# 19-1378), 0-9% zeros; DAPI off/on-nucleus 0.16-0.72 (NB: measured on the cycle-1 "DAPI", which
# on that slide is mostly autofluorescence -- see DAPI_BRIGHT; the real stain is 0.03-0.06). At E_PER_UNIT 120 and ADU_PER_E 0.5 one unit
# is ~60 ADU in solid tissue. The ranges below are deliberately WIDER than that calibration (a
# decade-plus either side), so other tissues -- less blue AF, cleaner or dirtier stains -- are inside
# the prior, not extrapolated. VERIFY: the centre comes from one slide, one tissue (heart).
TISSUE_BG = dict(marker=(0.1, 60.0), dapi=(0.3, 60.0))
# The counterstain's brightness is mapped from its theta level onto its OWN range instead of the
# antibody range (0.3-90 after EXPR_GAIN): drawn from the antibody range DAPI saturated the 16-bit
# frame (synthetic tissue median 8-16k ADU, p99.5 = 65535) against real nuclei at ~2-4k ADU.
# The theta level stays the same number -- only the rendered brightness is remapped, log-affinely,
# so the ordering of DAPI levels across types is preserved.
# RECALIBRATED 2026-11-10 against the right channel. The first calibration (0.3, 2.0) was read off
# channel 0 of the Slide2 backsub image, which markers_bs.csv names "DAPI" (cycle 1) but which on the
# Slide1-3 runs is mostly blue tissue autofluorescence -- nuclei ~1-2k ADU, off/on-nucleus 0.6-0.7.
# The real nuclear stain there is the re-stain "DAPI_cycle5": nuclei 24-44k ADU, tissue between
# nuclei 0.4-1.4k, off/on 0.03-0.06, up to ~0.6% saturated pixels. Across 13 slides the usable
# nuclear channel puts nuclei at 0.8-31k ADU (off/on 0.02-0.74, the weak end being those cycle-1
# stains). Measured on wide-prior renders (Otsu on both sides): DAPI_BRIGHT 2 -> nuclei ~1-4k,
# 8 -> ~2-18k, 30 -> ~4-64k with up to 6% saturation. (0.15, 40.0) spans the weak cycle-1 look up to
# a saturating stain. Synthetic off/on is 0.003-0.014 at the same estimator -- slightly SHARPER than
# real; the DAPI tissue_bg range (TISSUE_BG["dapi"]) is what lifts the floor. VERIFY on more tissues.
# 2026-11-16: (10, 800), same target, new units. The level used to be the CELL mean, so the nucleus
# itself was brighter by 1 / NUC_FRAC^3 (x2 at 0.8, x600 at 0.12) -- which is how 40 reached 64k ADU.
# It is now the brightness of the nucleus itself (render.loc_norm): 10-800 units x 60 ADU = 0.6k-48k
# ADU, the measured span of real nuclei, whatever the nucleus fraction. A tile draws one centre in
# that range and its types sit within DAPI_TYPE_DECADES of it (real nuclei differ ~10x inside one
# tile, not the 80x two independent draws gave). DAPI_CONTRAST is the counterstain's tissue
# background as a fraction of that centre: 0.01 is a clean re-stain (real off/on 0.02-0.06); 0.5,
# with the background's own texture on top, reaches the cycle-1 stain that is mostly blue
# autofluorescence (real 0.6-0.74). At 0.8 the texture peaks simply buried the nuclei.
DAPI_BRIGHT = (10.0, 800.0)
DAPI_TYPE_DECADES = 1.0
DAPI_CONTRAST = (0.01, 0.5)

# WHICH prior build_flexible_world draws from. "wide" (the default) is everything below on top of
# the calibrated generator; "run12" reproduces, bit-for-bit, the distribution the run12 checkpoint
# was trained on (no tissue background, DAPI on the antibody range, 100 ADU DAPI pedestal, the old
# DENSITY / ELONG / panel ranges, default textures, one fixed detector) -- the honest in-distribution
# control for that checkpoint. Every "wide" draw comes AFTER all run12 draws, so switching the prior
# never shifts the random stream of the parts both share.
PRIOR = "wide"
WIDE_PRIOR = dict(
    # density: sparser and denser than DENSITY, and down to near-empty tissue cover. MIN_DIST is now
    # only the empty-tile fallback -- a tissue tile's spacing follows its cell size (below).
    MIN_DIST=(3.0, 40.0), COVER=(0.05, 1.0), GROW=(1.0, 3.0),
    # cell SIZE, per tile (2026-11-10). The packing gives each pixel to the nearest seed in units of
    # that cell's own boundary, so a cell can never be bigger than its spacing allows: with MIN_DIST
    # drawn on its own (log-uniform 3-40 vox, median ~11 vox = 3.6 um) the realised cells had a median
    # section diameter of 3.2 um and the nuclei 3.4 um (Otsu 2.3 um), against real myocarditis nuclei
    # at ~5.3 um (Otsu, DAPI_cycle5) -- every real cell was ~2.3x bigger than anything trained on.
    # Now one tile scale is drawn log-uniform over CELL_RADIUS (vox, = the class RADIUS bounds, so
    # theta normalisation is unchanged: 1.3-15.6 um equivalent-sphere radius, lymphocyte to
    # cardiomyocyte), the spacing is that scale x SPACING_RATIO (log-uniform; ~1 = squeezed, confluent,
    # ~3 = loose), clipped to the class MIN_DIST bounds, and every root type's RADIUS is the tile scale
    # x exp(TYPE_RADIUS_SD x N(0,1)) -- so a tile can still mix small and big types around its scale.
    # 2026-11-16: the tile scale starts at 8 vox (2.6 um radius, a 5 um cell -- nothing nucleated is
    # smaller); the class RADIUS bounds, and so the theta normalisation, are unchanged.
    CELL_RADIUS=(8.0, 48.0), SPACING_RATIO=(1.0, 3.0), TYPE_RADIUS_SD=0.35,
    # CONFLUENCE is drawn as a regime, not left to chance. Measured on the run13 eval set the old
    # independent draws (spacing 1-3 R, GROW 1-3) left a median 30% of the frame labelled and most
    # cells as free blobs on black; real tissue tiles are ~100% tissue. A CONFLUENT tile (share
    # p_confluent) draws spacing = R x CONFLUENT_RATIO and GROW >= GROW_FILL x that ratio, which is
    # what fills the gaps (a Voronoi cell's far corner is ~2 seed spacings out); the rest are LOOSE
    # (the old draw: free shapes with gaps). MIN_SPACING keeps the densest tile under the query cap:
    # 9 vox = 2.9 um spacing is ~130 visible cells in a 52 um frame.
    p_confluent=0.65, CONFLUENT_RATIO=(0.7, 1.4), GROW_FILL=2.0, p_full_cover=0.6, MIN_SPACING=9.0,
    # mixed sizes: a root type other than the first ignores the tile scale with this probability
    # (radius log-uniform over CELL_RADIUS), and the spacing follows each cell's own radius by
    # TissueGeometry.SIZE_COUPLE -- big cells with small interstitial ones between them
    p_free_radius=0.3, SIZE_COUPLE=(0.3, 1.0),
    # the section is cut from bulk: TissueGeometry.SLAB = block + 2 x SLAB_REACH x the longest
    # semi-axis of the biggest type, capped (candidates are finite: at SLAB_MAX only cells of
    # radius >~ 20 vox still pack to saturation, which is exactly when a slab that deep is asked for)
    SLAB_REACH=1.5, SLAB_MAX=400.0,
    # cell shape. ELONG is now the TRUE long:short ratio (scene._stretch): 40% of types near-round,
    # the rest log-uniform up to 10 (a 4.6 R x 0.46 R rod); a few oblate
    ELONG=(0.5, 10.0), p_round=0.4, ROUND_ELONG=(0.8, 1.6),
    # the nucleus is sized in MICRONS and NUC_FRAC derived from it: drawn as a fraction of the cell
    # (0.1-0.8 of 1.3-15.6 um) nuclei ran from 0.13 to 12 um radius. 1.6-5 um spans a lymphocyte
    # to a large cardiomyocyte or tumour nucleus, so a big cell gets a SMALL nucleus fraction.
    NUC_RADIUS_UM=(1.6, 5.0), NUC_ELONG=(1.0, 3.2), p_nuc_round=0.3, NUC_ROUGH=(0.02, 0.35),
    NUC_BETA=(1.5, 5.0), RIM=(0.0, 1.5),
    # grain: with p_axis a tile has one direction every aligned type follows (in plane or through
    # it), strength AXIS_ALIGN; a type's own ALIGN is 1 with p_type_aligned, else uniform
    p_axis=0.5, AXIS_ALIGN=(0.5, 1.0), p_type_aligned=0.5,
    # panel content. The share of NUCLEAR markers is drawn per tile: with mu uniform over the tau
    # axis 42% of every panel was nuclear, so a nucleus showed in a dozen channels and the model
    # never needed to know which one was the counterstain; a real panel has 2-3 of 47. A marker is
    # UBIQUITOUS (positive in most types -- organelle and structural markers) with p_ubiquitous,
    # otherwise each type is positive with a per-marker probability from P_POS. p_ecm is the
    # per-tile share of EXTRACELLULAR markers (no type expresses them; they light the matrix).
    p_nuclear=(0.0, 0.5), p_ubiquitous=0.12, P_POS=(0.05, 0.45), UBIQ_POS=(0.7, 1.0),
    p_ecm=(0.0, 0.15),
    # within-type spread, polarity, band width
    CELL_CV=(0.0, 0.7), DAPI_CELL_CV=(0.0, 0.7), p_polar=0.25, POLARITY=(-1.5, 1.5),
    BAND_WIDTH_SD=0.5, BAND_SHARP_SD=0.3,
    # the shared background: n_bg hidden patterns every channel mixes (see
    # flexible_types.sample_background), a type's level on each lognormal with BG_TYPE_SD, and
    # n_matrix extracellular textures. MATRIX_REL is the matrix brightness of an ordinary channel
    # relative to its cell background; ECM_LEVEL the matrix brightness of an extracellular marker.
    # BG_TEXTURE_S caps the shared patterns' texture strength: at the antibody range (up to 2,
    # exp(2 z)) they were a field of hot blobs, where real unspecific signal is a mild relief.
    BG_TEXTURE_S=(0.15, 0.9),
    n_bg=(1, 3), BG_TYPE_SD=0.7, BG_CELL_CV=(0.0, 0.5), n_matrix=(1, 3), MATRIX_REL=(0.03, 2.0),
    ECM_LEVEL=(1.0, 60.0), p_peri=0.5, PERI=(0.3, 1.5), PERI_UM=(0.3, 1.5),
    # optics, per tile. The objective is not known (# VERIFY): the MACSima literature says 20x/0.75,
    # the first calibration here assumed 0.45, so NA is a nuisance drawn from a bank (a bank, not a
    # range, so the PSF kernels cache). FOCUS_UM is where the focal plane sits relative to the
    # section centre; most tiles are in focus.
    NA=(0.45, 0.55, 0.65, 0.75, 0.85, 0.95), FOCUS_UM=(-1.5, -0.75, 0.0, 0.0, 0.0, 0.0, 0.75, 1.5),
    # panel: the real slide has 48 channels, beyond the old 40 ceiling; background-only share per tile
    m_range=(6, 56), p_background=(0.0, 0.6),
    # marker texture: each antibody component's texture scalars uniform over their CLASS bounds (they
    # were never sampled -- every marker of a kind had the default texture); `s` floored so a texture
    # never vanishes outright, `w` of the extra components so multi-component mixes vary too
    # 2026-11-16: coherence / lam / length too, so Network and Sheet reach the crisp reticular and
    # striated looks (coherence -> 1 is a narrow-band, periodic field) real organelle markers have
    texture_keys=("scale", "clust", "fill", "soft", "coherence", "lam", "length"),
    texture_log_keys=("scale", "clust", "lam", "length"),
    texture_s=(0.2, 2.0), texture_w=(0.2, 1.0),
    # the camera / staining run, per tile: everything the instrument and the backsub do differently
    # from slide to slide. Not theta (DET is pinned) -- nuisance the model must be invariant to.
    DET=dict(E_PER_UNIT=(60.0, 300.0), READ_E=(8.0, 40.0), DARK_E=(0.0, 20.0),
             AF_SCALE_UM=(5.0, 60.0), AF_CV=(0.1, 0.9), ILLUM_CV=(0.0, 0.15), VIGNETTE=(0.0, 0.35),
             BG_DRIFT=(0.0, 0.25), BG_SHIFT_PX=(0.0, 2.5), BG_MARGIN_ADU=(0.0, 15.0),
             HOT_RATE=(0.0, 3e-4), READ_SMOOTH_PX=(0.0, 1.0), TBG_TEX_UM=(0.5, 4.0),
             TBG_TEX_CV=(0.1, 1.0), DAPI_OFFSET_ADU=(0.0, 120.0)),
)
# the run12 values the "run12" prior restores (kept here so the switch is data, not code)
RUN12_PRIOR = dict(ELONG=(1.0, 7.0))
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
        p_background=dict(start=0.1,      final=0.3),       # fraction of background-only markers
    ),
)


# DENSITY is the per-tile cellularity/structure spread, drawn fresh for EVERY tile in
# build_flexible_world -- deliberately NOT a curriculum (no start/final): the model must handle the
# whole span at every training round, from a fully-packed field with no empty background down to a
# genuinely empty patch. MIN_DIST/COVER/GROW are voxels / fraction / confluence as on TissueGeometry;
# max_structures is the inclusive ceiling on structure instances (0 is now reachable -> a cell-only
# tile); p_empty is the share of tiles that carry ZERO cells (COVER 0, no structures) as negatives.
DENSITY = dict(
    MIN_DIST=(4.0, 30.0),      # vox: 4 ~ fully packed (no gaps), 30 ~ sparse (a few cells). Cell
                               # count ~ 1/MIN_DIST^2, so the very-sparse end (>30) adds almost
                               # nothing between "sparse" and the explicit empty tiles below.
    COVER=(0.15, 1.0),         # fraction of the frame that is tissue; 1.0 = no empty background
    GROW=(1.1, 2.8),           # confluence: ~1.1 gaps between cells, ~2.8 confluent sheet
    max_structures=3,          # structure count drawn uniform in [0, max_structures]
    p_empty=0.10,              # ~10% genuinely empty (zero-cell) tiles
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
        p_background=float(c["p_background"]["start"]
                           + t * (c["p_background"]["final"] - c["p_background"]["start"])),
    )


# ============================================================ amortized inversion
# sigma in [2.5e-3, 7.4]; theta lives in [0, 1]
LOGVAR_MIN, LOGVAR_MAX = -12.0, 4.0
