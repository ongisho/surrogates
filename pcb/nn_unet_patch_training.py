"""Uniform random patch sampling for continuous PCB regression, without augmentation.

Cases are sampled with replacement. Crop all input channels and the target at
the same original-grid location. This is a regression sampler, not nnU-Net's
segmentation loader or foreground-label oversampling pipeline.
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class RandomPatchDataset(Dataset):
    def __init__(self, image_dataset, patch_size=(128, 128), samples_per_epoch=4000, seed=42):
        if len(image_dataset) < 1:
            raise ValueError("The training image dataset must be nonempty")
        if len(patch_size) != 2 or any(not isinstance(v, int) or v < 1 for v in patch_size):
            raise ValueError("patch_size must contain two positive integers")
        if not isinstance(samples_per_epoch, int) or samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be a positive integer")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.image_dataset = image_dataset
        self.patch_size = tuple(patch_size)
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Index-based seeding is reproducible even if worker scheduling changes.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        case_index = int(rng.integers(len(self.image_dataset)))
        image, target = self.image_dataset[case_index]
        height, width = image.shape[-2:]
        ph, pw = self.patch_size
        if tuple(target.shape[-2:]) != (height, width):
            raise ValueError("Input and temperature grids must be aligned")
        if ph > height or pw > width:
            raise ValueError(f"Patch {self.patch_size} exceeds training image {(height, width)}")
        y = int(rng.integers(height - ph + 1))
        x = int(rng.integers(width - pw + 1))
        # No flips, rotations, interpolation, intensity changes or coordinate resets.
        return (image[:, y:y + ph, x:x + pw].contiguous(),
                target[:, y:y + ph, x:x + pw].contiguous())
