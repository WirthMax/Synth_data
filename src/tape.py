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
        self.a = self.rng.standard_normal(self.K)                      # boundary harmonics
        self.b = self.rng.standard_normal(self.K)
        self.noise = self.rng.standard_normal((Pool, self.size, self.size))  # texture field
        
        # Tissuetile=256, n_cand=1500
        self.xy=self.rng.random((n_cand, 2)) * tile      # candidate centres
        self.order=self.rng.random(n_cand)               # priority mark for the hard-core rule
        self.support=self.rng.random((tile, tile))        # support field for the tissue

    # Add this method to allow bracket access
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{key}' is not a valid attribute of Tape")