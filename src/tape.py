import numpy as np


class Tape(object):
    def __init__(self, seed, size=201, K=20):
        self.seed = seed
        self.size = size
        self.K = K
        
        # Initialize the generator directly (avoids method naming collision)
        self.rng = np.random.default_rng(seed=self.seed)
        self.draw()

    def draw(self):
        """Everything random, drawn once, before any parameter is looked at."""
        self.a = self.rng.standard_normal(self.K)                      # boundary harmonics
        self.b = self.rng.standard_normal(self.K)
        self.noise = self.rng.standard_normal((self.size, self.size))  # texture field

    # Add this method to allow bracket access
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{key}' is not a valid attribute of Tape")