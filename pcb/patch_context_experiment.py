"""Build controlled same-input/different-context pairs from existing simulations."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import combinations, zip_longest
from pathlib import Path
import numpy as np
import hashlib
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from dataset import PCBDataset
from nn_unet import PCBImageDataset

FIXED_FIELDS = ("instance_map", "copper_mask", "ceramic_mask", "baseplate_mask", "h_normalized")


@lru_cache(maxsize=16)
def read_case(path):
    with h5py.File(path, "r") as f:
        fields = {key: f[key][:] for key in (*FIXED_FIELDS, "power_map", "temperature")}
    if not all(np.isfinite(value).all() for value in fields.values()):
        raise ValueError(f"Nonfinite data in {path}")
    return fields


def box_sums(array, size):
    integral = np.pad(array.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]


def pair_record(root, a, b, size=64, margin=16):
    first, second = read_case(str(root / a["path"])), read_case(str(root / b["path"]))
    if not all(np.array_equal(first[k], second[k]) for k in FIXED_FIELDS):
        return None
    shape = first["power_map"].shape
    if shape != second["power_map"].shape or len(shape) != 2 or min(shape) < size:
        return None
    if first["temperature"].shape != shape or second["temperature"].shape != shape:
        raise ValueError("Temperature and power grids must be aligned")
    changed = first["power_map"] != second["power_map"]
    if not changed.any():
        return None
    # Exclude changed power from the patch AND a surrounding margin.
    expanded_changes = box_sums(np.pad(changed, margin), size + 2 * margin)
    powered = box_sums(first["power_map"] != 0, size)
    valid = (expanded_changes == 0) & (powered > 0)
    if not valid.any():
        return None
    # Input-only choice: maximize unchanged local powered pixels; deterministic ties.
    score = np.where(valid, powered, -1)
    y, x = np.unravel_index(score.argmax(), score.shape)
    region = np.s_[y:y + size, x:x + size]
    assert np.array_equal(first["power_map"][region], second["power_map"][region])
    ta, tb = (case["temperature"][region].astype(np.float64) for case in (first, second))
    difference = float(np.linalg.norm(tb - ta))
    norm_a, norm_b = float(np.linalg.norm(ta)), float(np.linalg.norm(tb))
    if min(norm_a, norm_b) == 0:
        raise ValueError("Cannot report RelL2 with a zero-norm temperature target")
    ids = np.unique(first["instance_map"][changed])
    geometry_hash = hashlib.sha256()
    for field in FIXED_FIELDS[:-1]:
        geometry_hash.update(str((first[field].shape, first[field].dtype)).encode())
        geometry_hash.update(first[field].tobytes())
    return dict(topology=a["topology"], mask_id=str(a["mask_id"]), h_value=int(a["h_value"]),
                geometry_hash=geometry_hash.hexdigest(),
                path_a=a["path"], path_b=b["path"],
                Pigbt_a=int(a["Pigbt"]), Pigbt_b=int(b["Pigbt"]),
                Pfwd_a=int(a["Pfwd"]), Pfwd_b=int(b["Pfwd"]),
                row=int(y), column=int(x), patch_size=size, margin=margin,
                valid_powered_patch_locations=int(valid.sum()),
                changed_chip_ids=[int(i) for i in ids if i > 0],
                changed_background_pixels=int((changed & (first["instance_map"] == 0)).sum()),
                local_inputs_identical=True, geometry_identical=True, cooling_identical=True,
                target_difference_rel_l2=difference / norm_a,
                unavoidable_worst_case_rel_l2=difference / (norm_a + norm_b))


def build_matched_subset(root, n_pairs=6, patch_size=64, margin=16, max_layouts=100,
                         pairs_per_layout=1):
    """Verified input-controlled pairs, balanced across layouts before extra pairs.

    Change one power parameter at a time. No target-difference ranking is used.
    """
    root = Path(root)
    if (n_pairs is not None and n_pairs < 2) or patch_size < 1 or margin < 0 or pairs_per_layout < 1:
        raise ValueError("Invalid pair count, patch size, margin or per-layout limit")
    manifest = pd.read_csv(root / "metadata/manifest.csv", dtype={"mask_id": str})
    manifest = manifest[manifest["split"] == "TrainData"].copy()
    by_topology = defaultdict(list)
    for (topology, mask_id), group in manifest.groupby(["topology", "mask_id"], sort=True):
        by_topology[topology].append((topology, mask_id, group))
    ordered = [item for level in zip_longest(*by_topology.values()) for item in level if item is not None]
    layouts, seen_geometries = [], set()
    for index, (_, _, layout) in enumerate(ordered[:max_layouts], 1):
        found = []
        for _, conditions in layout.groupby("h_value", sort=True):
            rows = conditions.sort_values(["Pigbt", "Pfwd", "path"]).to_dict("records")
            for a, b in combinations(rows, 2):
                if (a["Pigbt"] == b["Pigbt"]) == (a["Pfwd"] == b["Pfwd"]):
                    continue
                record = pair_record(root, a, b, patch_size, margin)
                if record is None or record["geometry_hash"] in seen_geometries:
                    continue
                if found and record["geometry_hash"] != found[0]["geometry_hash"]:
                    continue
                found.append(record)
                if len(found) == pairs_per_layout:
                    break
            if len(found) == pairs_per_layout:
                break
        if found:
            layouts.append(found)
            seen_geometries.add(found[0]["geometry_hash"])
        if index % 25 == 0:
            print(f"Examined {index}/{len(ordered[:max_layouts])} layouts; "
                  f"verified {sum(map(len, layouts))} candidate pairs", flush=True)
        if n_pairs is not None and pairs_per_layout == 1 and len(layouts) == n_pairs:
            break
    # Include one pair from each layout before taking second/third pairs.
    selected = [r for level in zip_longest(*layouts) for r in level if r is not None]
    if n_pairs is not None:
        if len(selected) < n_pairs:
            raise ValueError(f"Only {len(selected)} verified pairs; increase pairs_per_layout or reduce n_pairs")
        selected = selected[:n_pairs]
    if len(selected) < 2:
        raise ValueError("Need at least two verified pairs")
    for i, record in enumerate(selected):
        record["pair_id"] = i
    print(f"Selected {len(selected)} verified pairs across "
          f"{len({r['geometry_hash'] for r in selected})} geometries", flush=True)
    return selected


def split_pairs(records, n_val=2, seed=42):
    """n_val counts whole geometries; every pair/member stays with its geometry."""
    geometries = list(dict.fromkeys(r["geometry_hash"] for r in records))
    if not 1 <= n_val < len(geometries):
        raise ValueError("Need at least one training and one validation geometry")
    order = np.random.default_rng(seed).permutation(len(geometries))
    val_hashes = {geometries[int(i)] for i in order[:n_val]}
    return [dict(r, subset="val" if r["geometry_hash"] in val_hashes else "train") for r in records]


class MatchedPatchDataset(Dataset):
    """Unique simulated patches; pair_indices maps each pair to its two samples.

    Reusing a baseline simulation in several pairs does not increase its weight.
    """
    def __init__(self, root, records, stats):
        self.records = records
        entries, lookup, self.pair_indices = [], {}, []
        for r in records:
            indices = []
            for side in ("path_a", "path_b"):
                key = (str(Path(root) / r[side]), r["row"], r["column"], r["patch_size"])
                if key not in lookup:
                    lookup[key] = len(entries)
                    entries.append(key)
                indices.append(lookup[key])
            self.pair_indices.append(tuple(indices))
        self.files = [Path(entry[0]) for entry in entries]
        images = PCBImageDataset(PCBDataset(root, "TrainData", stats, stride=1, files=self.files))
        self.patches = []
        for i, (_, y, x, size) in enumerate(entries):
            image, target = images[i]
            self.patches.append((image[:, y:y + size, x:x + size].contiguous(),
                                 target[:, y:y + size, x:x + size].contiguous()))
        for r, (a, b) in zip(records, self.pair_indices):
            if not torch.equal(self.patches[a][0], self.patches[b][0]):
                raise AssertionError(f"Pair {r['pair_id']} does not have identical model inputs")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        return self.patches[index]


def plot_pair(root, record):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    a, b = (read_case(str(Path(root) / record[key])) for key in ("path_a", "path_b"))
    y, x, size = record["row"], record["column"], record["patch_size"]
    region = np.s_[y:y + size, x:x + size]
    panels = [
        (a["power_map"], "Power A", "power"), (b["power_map"], "Power B", "power"),
        (b["power_map"] - a["power_map"], "External power change B-A", None),
        (a["instance_map"], "Fixed chip instances", None),
        (a["temperature"], "Temperature A (C)", "temperature"),
        (b["temperature"], "Temperature B (C)", "temperature"),
        (b["temperature"] - a["temperature"], "Temperature change B-A (C)", None),
        (a["copper_mask"], "Fixed copper mask", None),
        (a["power_map"][region], "Identical local power", "power"),
        (a["temperature"][region], "Local target A (C)", "temperature"),
        (b["temperature"][region], "Local target B (C)", "temperature"),
        (b["temperature"][region] - a["temperature"][region], "Local temperature change (C)", None),
    ]
    limits = {key: (min(a[field].min(), b[field].min()), max(a[field].max(), b[field].max()))
              for key, field in (("power", "power_map"), ("temperature", "temperature"))}
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for i, (ax, (array, title, scale)) in enumerate(zip(axes.flat, panels)):
        kwargs = dict(zip(("vmin", "vmax"), limits[scale])) if scale else {}
        im = ax.imshow(array, **kwargs)
        if i < 8:
            ax.add_patch(Rectangle((x - .5, y - .5), size, size, fill=False, edgecolor="red", linewidth=1.5))
        ax.set_title(title); ax.set_axis_off(); fig.colorbar(im, ax=ax, shrink=.7)
    fig.suptitle(f"Pair {record['pair_id']} | {record['subset']} | {record['topology']}/{record['mask_id']} | "
                 f"target-to-target RelL2={record['target_difference_rel_l2']:.4f}")
    fig.tight_layout()
    return fig
