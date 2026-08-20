@dataclass(frozen=True)
class Config:
    DYES = ("FITC", "PE", "APC")
    DAPI_NAME = "DAPI"
    DAPI_DYE = "DAPI"
    NUCLEAR_TAU_MAX = 0.0
    POOL_BANK = 0
    KEEP_DEFAULT = [r"\.scale$", r"\.clust$", r"\.fill$", r"\.s$", r"\.amp$"]
        
    SEED = None
    N_CAND = 1500
    SIZE = 201
    TILE = 256
    K = 20

    # lowest SH degree. 0 = pure size (absorbed by the volume normalisation),
    # 1 = shifts the centroid off the seed. 2 = lowest true shape mode.
    L_MIN = 2
    # 4 is the ceiling of the hardcoded Cartesian forms
    L = 4

    # microns per LATERAL voxel
    UM_PER_VOX = 0.325
    # z voxels this many times COARSER than lateral
    Z_RATIO = 1.0
    # (sz, sy, sx) = voxel SIZE per axis, in lateral units
    SPACING = (Z_RATIO, 1.0, 1.0)   
    VOL = (128, 128, 128)
    IMG = (128, 128, 3)
    TISSUE_VOL = (40, 160, 160)
    
    DET_PARAMS = {
        "E_PER_UNIT": 120.0,
        "READ_E": 2.5,
        "DARK_E": 5.0,
        "ADU_PER_E": 0.5,
        "OFFSET_ADU": 100.0,
        "AF_SCALE_UM": 25.0,
        "AF_CV": 0.45,
        "ILLUM_CV": 0.06,
        "VIGNETTE": 0.18
    }