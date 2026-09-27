from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def load_elasticity_data(xy_path: str, stress_path: str):
    """
    Load the Transolver elasticity benchmark into CPU RAM.

    Expected raw file layouts:
        XY:     [N, 2, num_samples]
        stress: [N, num_samples]

    Returns:
        xy:     float32 tensor [num_samples, N, 2]
        stress: float32 tensor [num_samples, N]
    """
    xy = torch.from_numpy(np.load(xy_path)).float().permute(2, 0, 1).contiguous()
    stress = torch.from_numpy(np.load(stress_path)).float().permute(1, 0).contiguous()

    if xy.shape[0] != stress.shape[0] or xy.shape[1] != stress.shape[1]:
        raise ValueError(
            f"Incompatible shapes after loading: xy={tuple(xy.shape)}, "
            f"stress={tuple(stress.shape)}"
        )

    return xy, stress


class OutputNormalizer:
    """
    Global standardization of the output stress:
        y_norm = (y - mean) / std

    Fit ONLY on training targets to avoid test-data leakage.

    This matches the elasticity use of Transolver's UnitTransformer:
    mean/std are computed across samples and mesh points.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    @classmethod
    def fit(cls, y_train: torch.Tensor, eps: float = 1e-8):
        mean = y_train.mean()
        std = y_train.std() + eps
        return cls(mean, std)

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean.to(y.device)) / self.std.to(y.device)

    def decode(self, y_normalized: torch.Tensor) -> torch.Tensor:
        return y_normalized * self.std.to(y_normalized.device) + self.mean.to(
            y_normalized.device
        )

    def __repr__(self):
        return (
            f"OutputNormalizer(mean={self.mean.item():.6g}, "
            f"std={self.std.item():.6g})"
        )


class ElasticityDataset(Dataset):
    """
    Dataset for the Transolver elasticity benchmark.

    Inputs:
        x = XY coordinates, shape [N, 2]

    Targets:
        y = stress, shape [N, 1]

    The XY coordinates are left unchanged.
    Only the output stress is optionally normalized.
    """

    def __init__(
        self,
        xy: torch.Tensor,
        stress: torch.Tensor,
        indices,
        output_normalizer: OutputNormalizer | None = None,
        normalize_output: bool = True,
    ):
        self.xy = xy
        self.stress = stress
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.output_normalizer = output_normalizer
        self.normalize_output = normalize_output

        if self.normalize_output and self.output_normalizer is None:
            raise ValueError(
                "output_normalizer is required when normalize_output=True."
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        x = self.xy[idx]                     # [N, 2]
        y = self.stress[idx].unsqueeze(-1)  # [N, 1]

        if self.normalize_output:
            y = self.output_normalizer.encode(y)

        return x, y
