import numpy as np
from scene import n_sh


class Tape(object):
    def __init__(self, seed, size=201, K=20, n_ch_max = 20):
        self.seed = seed
        self.size = size
        self.K = K
        self.n_ch_max = n_ch_max
        
        # Initialize the generator directly (avoids method naming collision)
        self.rng = np.random.default_rng(seed=self.seed)
        self.draw()

    def draw(self, tile=256, n_cand=1500, Pool = 3, ):
        """Everything random, drawn once, before any parameter is looked at."""
        self.tile = tile
        ### Cytoplasmic
        # boundary harmonics
        self.a = self.rng.standard_normal((n_cand, self.K))
        self.b = self.rng.standard_normal((n_cand, self.K))
        # texture field (sandbox resolution, one plane per pool)
        self.noise = self.rng.standard_normal((Pool, self.size, self.size))
        # texture field at tile resolution -- sliced per patch when rendering tissue markers
        self.noise_tile = self.rng.standard_normal((Pool, tile, tile))
        ### Nuclear
        # independent nuclear dynamics
        self.a2 = self.rng.standard_normal((n_cand, self.K))
        self.b2 = self.rng.standard_normal((n_cand, self.K))
        # per-cell nucleus:cell size ratio
        self.z_nucfrac = self.rng.standard_normal(n_cand)
        # nucleus offset direction
        self.u_offdir = self.rng.random(n_cand)
        # nucleus offset magnitude
        self.u_offmag = self.rng.random(n_cand)
        
        # ### Tissue
        # candidate centres
        self.xy=self.rng.random((n_cand, 2)) * tile
        # priority mark for the hard-core rule
        self.order=self.rng.random(n_cand)
        # support field for the tissue
        self.support=self.rng.random((tile, tile))       
        # per-cell size jitter
        self.z_size=self.rng.standard_normal(n_cand)
        # per-cell orientation
        self.u_orient=self.rng.random(n_cand)
        
        
    def draw3d(self, vol=(128, 128, 128), n_cand=1500, Pool = 3, l_min = 2, L = 4):
        self.vol = tuple(vol)
        self.L = L
        self.l_min = l_min
        self.n_lm = n_sh(L, l_min)
        # Every degree l carries orders m = −l, −l+1, …, +l, which is 2l + 1 functions.
        # cellular harmoics
        self.sh = self.rng.standard_normal((n_cand, self.n_lm))
        # nuclear harmoics
        self.sh2 = self.rng.standard_normal((n_cand, self.n_lm))
        # Noise for 3D case
        self.texture_noise = self.rng.standard_normal((Pool, *self.vol)).astype(np.float32)
        self.gate_noise = self.rng.standard_normal((Pool, *self.vol)).astype(np.float32)
        
    def drawSensor(self, shape=(128, 128, 3)):
        ny, nx, nc = shape
        # photon shot noise
        self.z_shot = self.rng.standard_normal((nc, ny, nx), dtype=np.float32)
        # camera read noise
        self.z_read = self.rng.standard_normal((nc, ny, nx), dtype=np.float32)
        # autofluorescence texture
        self.w_af   = self.rng.standard_normal((nc, ny, nx), dtype=np.float32)
        # flat field texture
        self.w_ill  = self.rng.standard_normal((ny, nx), dtype=np.float32)
        
    def drawTissue(self, shape=(128, 128, 3), n_cand=1500, n_lm = 21, Pool = 3):
        nz, ny, nx = shape
        v = self.rng.standard_normal((n_cand, 3))
        # centres, (x,y,z) in voxels
        self.xyz=self.rng.random((n_cand, 3)) * np.array([nx, ny, nz])
        # hard-core priority
        self.order=self.rng.random(n_cand)
        # tissue support field
        self.support3=self.rng.standard_normal(shape, dtype=np.float32)
        # polar / azim / roll
        self.u_orient3=self.rng.random((n_cand, 3))
        # which cell type
        self.u_type=self.rng.random(n_cand)
        self.z_size=self.rng.standard_normal(n_cand)
        self.z_nucfrac=self.rng.standard_normal(n_cand)
        # a DIRECTION needs three numbers -- one scalar normalises to 1.0 and every nucleus
        # then displaces along the same body-frame diagonal
        self.u_offdir=v / np.linalg.norm(v, axis=1, keepdims=True)
        self.u_offmag=self.rng.random(n_cand)
        self.sh=self.rng.standard_normal((n_cand, n_lm))
        self.sh2=self.rng.standard_normal((n_cand, n_lm))
        self.texture_noise=self.rng.standard_normal((Pool, *shape), dtype=np.float32)
        self.gate_noise=self.rng.standard_normal((Pool, *shape), dtype=np.float32)


    # Add this method to allow bracket access
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{key}' is not a valid attribute of Tape")