"""v3 dataset: reads flexible-mode bundle shards (one npz per sample, written by run_v3.py gen /
flexible_bundle.build_bundle) and turns each into the loss target dict via targets_v3. synmodel-side
(numpy + torch, no scipy). Samples have different M and J; `targets_v3.collate_v3` batches them.
"""
import glob
import os

import numpy as np
import torch

import targets_v3 as T


class BundleShardDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, k_slots, n_global):
        self.paths = sorted(glob.glob(os.path.join(data_dir, "bundle_*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"no bundle_*.npz under {data_dir}")
        self.k_slots, self.n_global = k_slots, n_global

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        b = dict(np.load(self.paths[i], allow_pickle=True))
        return T.build_sample_targets(b, k_slots=self.k_slots, n_global=self.n_global)


def make_loader(data_dir, k_slots, n_global, batch_size, shuffle, num_workers=0, m_max=None):
    ds = BundleShardDataset(data_dir, k_slots, n_global)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=lambda s: T.collate_v3(s, m_max=m_max), drop_last=False)
