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
    def __init__(self, data_dir, k_slots, n_global, augment=False, chan_drop=0.0):
        # several dirs may be joined with commas -- the open-ended rounds train on a rolling pool of
        # the last few rounds, so a slow generator does not mean many epochs over one round's data
        self.paths = sorted(p for d in str(data_dir).split(",") if d
                            for p in glob.glob(os.path.join(d, "bundle_*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"no bundle_*.npz under {data_dir}")
        self.k_slots, self.n_global = k_slots, n_global
        # augment: a random flip + quarter turn per sample (targets_v3.augment_bundle). Drawn with
        # torch, which DataLoader seeds per worker (numpy's global RNG would repeat across workers).
        self.augment = augment
        # chan_drop: probability that a training sample is shown with a random SUB-PANEL -- each
        # antibody channel kept with one probability drawn in [0.5, 1], the counterstain always,
        # at least 4 channels (targets_v3.drop_channels)
        self.chan_drop = float(chan_drop)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        b = dict(np.load(self.paths[i], allow_pickle=True))
        if self.augment:
            k_rot, flip = int(torch.randint(4, ())), bool(torch.randint(2, ()))
            b = T.augment_bundle(b, k_rot=k_rot, flip=flip)
        if self.chan_drop > 0 and float(torch.rand(())) < self.chan_drop:
            M = int(b["th_M"])
            order = np.asarray(b["th_channel_order"]) if "th_channel_order" in b else np.arange(M)
            keep = (torch.rand(M) < float(0.5 + 0.5 * torch.rand(()))).numpy()
            keep[order == 0] = True                              # the counterstain stays
            short = min(M, 4) - int(keep.sum())
            if short > 0:
                off = np.flatnonzero(~keep)
                keep[off[torch.randperm(len(off))[:short].numpy()]] = True
            b = T.drop_channels(b, keep)
        return T.build_sample_targets(b, k_slots=self.k_slots, n_global=self.n_global)


def make_loader(data_dir, k_slots, n_global, batch_size, shuffle, num_workers=0, m_max=None):
    ds = BundleShardDataset(data_dir, k_slots, n_global)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=lambda s: T.collate_v3(s, m_max=m_max), drop_last=False)
