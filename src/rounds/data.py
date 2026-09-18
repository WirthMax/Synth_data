"""Shared dataset: reads the shard_*.images.npy/shard_*.npz pairs generate_samples writes,
across one or more directories (a range of round dirs for training, or a single eval-set dir).
Reuses model.to_input as the ONE preprocessing function, per spec, shared by training,
evaluation and any future real-data inference. synmodel-side only (needs torch)."""
import _pathfix  # noqa: F401

import glob
import os

import numpy as np
import torch

from model import to_input


class ShardedThetaImageDataset(torch.utils.data.Dataset):
    def __init__(self, dirs, augment=False, augment_seed=0):
        self.dirs = [dirs] if isinstance(dirs, str) else list(dirs)
        self._shards = []
        thetas, comp_ids = [], []
        self.shard_paths = []
        for d in self.dirs:
            for npz_path in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
                img_path = npz_path[: -len(".npz")] + ".images.npy"
                if not os.path.exists(img_path):
                    continue
                z = np.load(npz_path)
                u = z["u"]
                img = np.load(img_path, mmap_mode="r")
                if img.shape[0] != u.shape[0]:
                    raise ValueError(
                        f"{img_path}: {img.shape[0]} images but {u.shape[0]} theta rows in "
                        f"{npz_path}"
                    )
                self._shards.append(img)
                self.shard_paths.append(npz_path)
                thetas.append(u)
                comp_ids.append(z["component_id"] if "component_id" in z.files
                               else np.zeros(len(u), dtype=np.int32))
        if not thetas:
            raise FileNotFoundError(f"no shard_*.npz/.images.npy pairs found under {self.dirs}")
        self.theta = np.concatenate(thetas, axis=0).astype(np.float32)
        self.component_id = np.concatenate(comp_ids, axis=0).astype(np.int32)
        self._index = [(si, pos) for si, u in enumerate(thetas) for pos in range(len(u))]
        self.augment = augment
        self.augment_seed = augment_seed

    def __len__(self):
        return len(self.theta)

    @property
    def in_channels(self):
        return self._shards[0].shape[-1]

    def __getitem__(self, k):
        si, pos = self._index[k]
        a = np.asarray(self._shards[si][pos])
        if self.augment:
            # a fresh rng per (augment_seed, k), not one advanced call-to-call -- see the same
            # fix applied to model.MMAPView for why (reproducible regardless of shuffle order,
            # epoch, or DataLoader worker count).
            rng = np.random.default_rng(self.augment_seed + k)
            if rng.random() < 0.5:
                a = a[::-1]
            if rng.random() < 0.5:
                a = a[:, ::-1]
            r = int(rng.integers(4))
            if r:
                a = np.rot90(a, r, axes=(0, 1))
            a = np.ascontiguousarray(a)
        return torch.from_numpy(to_input(a)).permute(2, 0, 1), torch.from_numpy(self.theta[k])
