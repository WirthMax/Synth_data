import numpy as np


class Tape(object):
    def __init__(self, seed, size=201, K=20):
        self.seed = seed
        self.size = size
        self.K = K
        
        # Initialize the generator directly (avoids method naming collision)
        self.rng = np.random.default_rng(seed=self.seed)
        self.draw()

    def draw(self, tile=256, n_cand=1500, Pool = 3, ):
        """Everything random, drawn once, before any parameter is looked at."""
        self.tile = tile
        # boundary harmonics
        self.a = self.rng.standard_normal((n_cand, self.K))
        self.b = self.rng.standard_normal((n_cand, self.K))
        self.noise = self.rng.standard_normal((Pool, self.size, self.size))  # texture field
        
        # Tissuetile=256, n_cand=1500
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

    # Add this method to allow bracket access
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{key}' is not a valid attribute of Tape")