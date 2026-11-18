"""Whole-board context with a temperature prediction only for the requested patch.

The local encoder/decoder and global residual encoder come from the installed
nnU-Net architecture package. Token mixing, region sampling, and fusion are our
2D regression adapter, not nnU-Net's built-in 3D segmentation cascade.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from dataset import PCBDataset
from nn_unet import NNUNetRegressor, PCBImageDataset
from patch_context_experiment import MatchedPatchDataset


class GlobalContextPatchDataset(MatchedPatchDataset):
    """Same unique patches/targets as the local baseline, plus coarse inputs.

    Inputs are a dictionary: local [8,P,P], global [8,G,G], roi [4]. The ROI is
    (left, top, right, bottom) in full-board edge coordinates normalized to [0,1].
    Coarse channels are area averages of the SAME eight normalized input maps.
    Temperature targets, simulation IDs, and power-setting labels are not inputs.
    """

    def __init__(self, root, records, stats, global_size=64):
        if not isinstance(global_size, int) or global_size < 1:
            raise ValueError("global_size must be a positive integer")
        super().__init__(root, records, stats)
        self.global_size = global_size
        locations = {}
        for record, indices in zip(records, self.pair_indices):
            for index in indices:
                location = (record["row"], record["column"], record["patch_size"])
                if index in locations and locations[index] != location:
                    raise AssertionError("Inconsistent patch locations")
                locations[index] = location

        images = PCBImageDataset(PCBDataset(root, "TrainData", stats, stride=1, files=self.files))
        self.global_images = {}
        self.rois = []
        shapes = {}
        print(f"Preparing whole-board {global_size}x{global_size} input maps...", flush=True)
        for index, path in enumerate(self.files):
            if path not in self.global_images:
                image, _ = images[index]
                height, width = image.shape[-2:]
                if global_size > min(height, width):
                    raise ValueError("global_size must not exceed the full-board dimensions")
                shapes[path] = (height, width)
                self.global_images[path] = F.interpolate(
                    image.unsqueeze(0), size=(global_size, global_size), mode="area",
                )[0].contiguous()
            height, width = shapes[path]
            row, column, size = locations[index]
            if not (0 <= row <= height - size and 0 <= column <= width - size):
                raise ValueError("Patch lies outside its full board")
            self.rois.append(torch.tensor(
                [column / width, row / height, (column + size) / width, (row + size) / height],
                dtype=torch.float32,
            ))
            if (index + 1) % 250 == 0:
                print(f"  prepared {index + 1}/{len(self.files)} patch inputs", flush=True)
        for a, b in self.pair_indices:
            if torch.equal(self.global_images[self.files[a]], self.global_images[self.files[b]]):
                raise AssertionError("Coarsening erased the external power difference in a pair")
            if not torch.equal(self.rois[a], self.rois[b]):
                raise AssertionError("Paired patches must refer to the same board region")

    def __getitem__(self, index):
        local, target = super().__getitem__(index)
        return {"local": local, "global": self.global_images[self.files[index]],
                "roi": self.rois[index]}, target


class GlobalContextNNUNetRegressor(nn.Module):
    """Encode the whole coarse board, sample context, decode ONLY one patch.

    Both encoders use features_per_stage. A residual self-attention block mixes
    all global bottleneck tokens before any region is sampled. Thus sampled
    context can depend on distant chips, not only on the corresponding crop.
    That region is fused with the local bottleneck; native U-Net skips preserve
    local detail. No global decoder or full-board temperature head is created.
    """

    def __init__(self, input_channels=8, out_channels=1,
                 features_per_stage=(32, 64, 128, 256, 320),
                 n_conv_per_stage=2, n_conv_per_stage_decoder=2,
                 architecture="resenc", n_blocks_per_stage=None,
                 global_size=64, global_attention_heads=4):
        super().__init__()
        if architecture != "resenc":
            raise ValueError("This experiment uses the residual-encoder architecture")
        self.divisor = 2 ** (len(features_per_stage) - 1)
        if not isinstance(global_size, int) or global_size < 2 * self.divisor or global_size % self.divisor:
            raise ValueError("global_size must be divisible by the encoder stride and leave >=2 bottleneck cells")
        channels = features_per_stage[-1]
        if global_attention_heads < 1 or channels % global_attention_heads:
            raise ValueError("Bottleneck channels must be divisible by global_attention_heads")
        self.input_channels = input_channels
        self.global_size = global_size
        self.network = NNUNetRegressor(
            input_channels=input_channels, out_channels=out_channels,
            features_per_stage=features_per_stage, n_conv_per_stage=n_conv_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            architecture=architecture, n_blocks_per_stage=n_blocks_per_stage,
        ).network
        # Copy only the upstream encoder STRUCTURE, then initialize independently.
        # Keeping no global decoder avoids computing any full-board temperature.
        self.global_encoder = deepcopy(self.network.encoder)
        self.global_encoder.return_skips = False
        self.global_encoder.apply(self.network.initialize)
        self.global_norm = nn.LayerNorm(channels)
        self.global_attention = nn.MultiheadAttention(
            channels, global_attention_heads, dropout=0.0, batch_first=True,
        )
        self.global_ff_norm = nn.LayerNorm(channels)
        self.global_ff = nn.Sequential(
            nn.Linear(channels, 2 * channels), nn.GELU(), nn.Linear(2 * channels, channels),
        )
        self.context_fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )

    @staticmethod
    def crop_context(context, roi, output_size):
        """Sample the ROI with pixel-center alignment; preserve x/y orientation."""
        height, width = output_size
        ys = (torch.arange(height, device=context.device, dtype=context.dtype) + 0.5) / height
        xs = (torch.arange(width, device=context.device, dtype=context.dtype) + 0.5) / width
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        left, top, right, bottom = roi.to(context).unbind(dim=1)
        gx = left[:, None, None] + (right - left)[:, None, None] * xx
        gy = top[:, None, None] + (bottom - top)[:, None, None] * yy
        grid = 2 * torch.stack((gx, gy), dim=-1) - 1
        return F.grid_sample(context, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def forward(self, inputs):
        local, coarse, roi = inputs["local"], inputs["global"], inputs["roi"]
        if local.ndim != 4 or local.shape[1] != self.input_channels:
            raise ValueError(f"Expected local [B,{self.input_channels},H,W]")
        if coarse.shape != (local.shape[0], self.input_channels, self.global_size, self.global_size):
            raise ValueError("Unexpected whole-board coarse input shape")
        if roi.shape != (local.shape[0], 4) or not torch.isfinite(roi).all():
            raise ValueError("Expected finite ROI coordinates [B,4]")
        if (roi < 0).any() or (roi > 1).any() or (roi[:, 2:] <= roi[:, :2]).any():
            raise ValueError("ROI must have positive extent within the full board")
        height, width = local.shape[-2:]
        if min(height, width) < 2 * self.divisor or height % self.divisor or width % self.divisor:
            raise ValueError("Local patch dimensions must be divisible by the encoder stride and >=2*stride")
        # Process the entire coarse board BEFORE extracting the requested region.
        context = self.global_encoder(coarse)
        tokens = context.flatten(2).transpose(1, 2)
        normalized = self.global_norm(tokens)
        tokens = tokens + self.global_attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + self.global_ff(self.global_ff_norm(tokens))
        context = tokens.transpose(1, 2).reshape_as(context)
        skips = self.network.encoder(local)
        sampled = self.crop_context(context, roi, skips[-1].shape[-2:])
        skips[-1] = self.context_fusion(torch.cat((skips[-1], sampled), dim=1))
        return self.network.decoder(skips)
