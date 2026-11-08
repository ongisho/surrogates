"""Lazy HDF5 loader for henkuailederen/Multi-topology-Dataset.

One item is (coordinates [N,2], features [N,6], temperature [N,1]).
Features: standardized power, chip occupancy, ceramic, baseplate, copper,
and the supplied h_normalized. Instance IDs are categorical, so only their
occupancy is used. No target-derived max_temps are supplied to the network.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import time
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = ("TrainData", "ValData", "TestData")
FEATURE_NAMES = ("power", "chip", "ceramic", "baseplate", "copper", "h_normalized")


def discover_files(root, split):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    files = sorted((Path(root) / split).rglob("data.h5"))
    if not files:
        raise FileNotFoundError(f"No data.h5 under {Path(root) / split}")
    return files


@dataclass
class NormalizationStats:
    power_mean: float
    power_std: float
    temperature_mean: float
    temperature_std: float

    @classmethod
    def fit(cls, train_files, max_files=None, seed=42, cache_path=None, progress_every=100):
        """Stream training fields, optionally using a reproducible file subset.

        Only accept official training paths to prevent accidental split leakage.
        None uses every file; max_files estimates moments from at most that many
        uniformly sampled files. Every pixel in each selected file is used.
        Cache identity includes selection, file paths, sizes and modification times.
        """
        train_files = sorted(map(Path, train_files))
        if not train_files or any("TrainData" not in p.parts for p in train_files):
            raise ValueError("Fit statistics using nonempty TrainData files only.")
        if max_files is not None:
            if not isinstance(max_files, int) or max_files < 1:
                raise ValueError("max_files must be a positive integer or None")
            if len(train_files) > max_files:
                indices = np.sort(np.random.default_rng(seed).choice(len(train_files), max_files, replace=False))
                train_files = [train_files[i] for i in indices]
        fingerprint = None
        if cache_path is not None:
            cache_path = Path(cache_path)
            identity = []
            for path in train_files:
                info = path.stat()
                identity.append((str(path.resolve()), info.st_size, info.st_mtime_ns))
            fingerprint = hashlib.sha256(json.dumps([1, identity]).encode()).hexdigest()
            if cache_path.is_file():
                try:
                    cached = json.loads(cache_path.read_text())
                    if cached["fingerprint"] == fingerprint:
                        stats = cls(**cached["stats"])
                        values = list(stats.to_dict().values())
                        if np.isfinite(values).all() and stats.power_std > 0 and stats.temperature_std > 0:
                            print(f"Loaded normalization cache: {cache_path}", flush=True)
                            return stats
                except (ValueError, TypeError, KeyError):
                    pass  # Recompute an obsolete or incomplete cache.
        print(f"Fitting normalization on {len(train_files)} training files...", flush=True)
        started = time.monotonic()
        counts = [0, 0]
        means = [0., 0.]
        m2s = [0., 0.]
        for index, path in enumerate(train_files, 1):
            with h5py.File(path, "r") as f:
                for i, key in enumerate(("power_map", "temperature")):
                    a = np.asarray(f[key], dtype=np.float64)
                    if not np.isfinite(a).all():
                        raise ValueError(f"Nonfinite {key} in {path}")
                    n, mean, var = a.size, float(a.mean()), float(a.var())
                    delta = mean - means[i]
                    total = counts[i] + n
                    m2s[i] += n * var + delta**2 * counts[i] * n / total
                    means[i] += delta * n / total
                    counts[i] = total
            if progress_every and (index % progress_every == 0 or index == len(train_files)):
                print(f"Normalization: {index}/{len(train_files)} files, "
                      f"{time.monotonic() - started:.1f}s", flush=True)
        stds = [max((v / n)**0.5, 1e-8) for v, n in zip(m2s, counts)]
        stats = cls(means[0], stds[0], means[1], stds[1])
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            temporary.write_text(json.dumps({"fingerprint": fingerprint,
                                             "n_files": len(train_files), "stats": stats.to_dict()}, indent=2))
            temporary.replace(cache_path)
        return stats

    def to_dict(self):
        return asdict(self)

    def decode_temperature(self, value):
        return value * self.temperature_std + self.temperature_mean


class PCBDataset(Dataset):
    def __init__(self, root, split, stats, stride=4, files=None):
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer")
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.files = discover_files(root, split) if files is None else list(map(Path, files))
        if not self.files or any(split not in p.parts for p in self.files):
            raise ValueError("Files must belong to the requested official split.")
        self.stats, self.stride = stats, stride

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Open inside each access: safe for multiprocessing DataLoader workers.
        with h5py.File(self.files[index], "r") as f:
            shape = f["temperature"].shape
            if len(shape) != 2:
                raise ValueError("Expected a 2D temperature field")
            s = self.stride
            def read(key):
                if f[key].shape != shape:
                    raise ValueError(f"Unaligned field {key} in {self.files[index]}")
                a = np.asarray(f[key][::s, ::s], dtype=np.float32)
                if not np.isfinite(a).all():
                    raise ValueError(f"Nonfinite field {key}")
                return a
            power = (read("power_map") - self.stats.power_mean) / self.stats.power_std
            chip = (read("instance_map") > 0).astype(np.float32)
            masks = [read(key) for key in ("ceramic_mask", "baseplate_mask", "copper_mask")]
            h = float(np.asarray(f["h_normalized"]).reshape(-1)[0])
            if not np.isfinite(h):
                raise ValueError("Nonfinite cooling coefficient")
            features = np.stack([power, chip, *masks, np.full_like(power, h)], axis=-1)
            target = (read("temperature") - self.stats.temperature_mean) / self.stats.temperature_std
        # Coordinate order is (x=column, y=row), consistently flattened row-major.
        yy, xx = np.meshgrid(np.linspace(0, 1, shape[0], dtype=np.float32)[::s],
                             np.linspace(0, 1, shape[1], dtype=np.float32)[::s], indexing="ij")
        coords = np.stack([xx, yy], axis=-1)
        return tuple(torch.from_numpy(np.ascontiguousarray(a.reshape(-1, c)))
                     for a, c in ((coords, 2), (features, 6), (target, 1)))
