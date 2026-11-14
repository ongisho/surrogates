"""Continuous-field inference using nnU-Net's tile placement and Gaussian weights.

This regression wrapper blends raw continuous predictions. It does not use
nnU-Net's segmentation export, softmax, argmax or postprocessing.
"""
from __future__ import annotations

from itertools import product
import torch
from torch.nn import functional as F
from nnunetv2.inference.sliding_window_prediction import (
    compute_gaussian, compute_steps_for_sliding_window,
)


@torch.no_grad()
def predict_sliding_window(model, image, patch_size=(128, 128),
                           tile_step_size=0.5, sw_batch_size=4, use_gaussian=True):
    """Return CPU float32 predictions [B,C_out,H,W] with overlap blending.

    Keep full images/accumulators on CPU and send only tile batches to the model.
    Coordinates and normalization must already be computed for the full image;
    they are cropped together with other feature channels, never reset per tile.
    Smaller images are replicate-padded, then predictions cropped to original size.
    Training/evaluation mode is restored, and no autograd graph is constructed.
    """
    if image.ndim != 4 or image.shape[0] < 1 or min(image.shape[-2:]) < 1:
        raise ValueError("Expected nonempty [B,C,H,W] images")
    if len(patch_size) != 2 or any(not isinstance(v, int) or v < 1 for v in patch_size):
        raise ValueError("patch_size must contain two positive integers")
    if not 0 < tile_step_size <= 1:
        raise ValueError("tile_step_size must be in (0, 1]")
    if not isinstance(sw_batch_size, int) or sw_batch_size < 1:
        raise ValueError("sw_batch_size must be a positive integer")
    patch_size = tuple(patch_size)
    height, width = image.shape[-2:]
    image = image.detach().cpu()
    pad_h, pad_w = max(0, patch_size[0] - height), max(0, patch_size[1] - width)
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
    padded_shape = tuple(image.shape[-2:])
    positions = list(product(*compute_steps_for_sliding_window(padded_shape, patch_size, tile_step_size)))
    weight = (compute_gaussian(patch_size, dtype=torch.float32, device=torch.device("cpu"))
              if use_gaussian else torch.ones(patch_size, dtype=torch.float32))
    denominator = torch.zeros(padded_shape, dtype=torch.float32)
    ph, pw = patch_size
    for y, x in positions:
        denominator[y:y + ph, x:x + pw] += weight
    if not torch.isfinite(denominator).all() or not (denominator > 0).all():
        raise FloatingPointError("Invalid sliding-window coverage/weights")

    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    was_training = model.training
    model.eval()
    results = []
    try:
        for sample in image:
            numerator = None
            for start in range(0, len(positions), sw_batch_size):
                chunk = positions[start:start + sw_batch_size]
                tiles = torch.stack([sample[:, y:y + ph, x:x + pw] for y, x in chunk]).to(device)
                prediction = model(tiles).detach().to(device="cpu", dtype=torch.float32)
                if prediction.ndim != 4 or prediction.shape[0] != len(chunk) or tuple(prediction.shape[-2:]) != patch_size:
                    raise ValueError("Model must return [tile_batch,C_out,patch_height,patch_width]")
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite sliding-window predictions")
                if numerator is None:
                    numerator = torch.zeros((prediction.shape[1], *padded_shape), dtype=torch.float32)
                for tile, (y, x) in zip(prediction, chunk):
                    numerator[:, y:y + ph, x:x + pw] += tile * weight
            results.append((numerator / denominator)[..., :height, :width])
    finally:
        model.train(was_training)
    return torch.stack(results)
