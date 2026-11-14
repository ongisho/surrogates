"""PCB regression adapter using the unmodified pip-installed nnU-Net network.

Install: pip install nnunetv2==2.7.0 dynamic-network-architectures==0.4.3

This uses nnU-Net's network factory, not its segmentation trainer, automatic
planner, preprocessing, augmentation or inference pipeline. Architecture settings
are explicit and manually selected. No upstream network code is copied here.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class PCBImageDataset(Dataset):
    """Reshape the existing PCB point dataset into aligned channel-first images.

    Channels: x, y, standardized power, chip occupancy, ceramic, baseplate,
    copper, h/10000. Targets are standardized temperature, without quantization.
    """

    def __init__(self, point_dataset):
        self.point_dataset = point_dataset
        self.files = point_dataset.files

    def __len__(self):
        return len(self.point_dataset)

    def __getitem__(self, index):
        coords, features, target = self.point_dataset[index]
        height = int(torch.unique(coords[:, 1]).numel())
        width = int(torch.unique(coords[:, 0]).numel())
        if height * width != coords.shape[0]:
            raise ValueError("Expected a complete rectangular grid in row-major order")
        inputs = torch.cat((coords, features), dim=-1)
        image = inputs.T.reshape(inputs.shape[-1], height, width).contiguous()
        temperature = target.T.reshape(target.shape[-1], height, width).contiguous()
        return image, temperature


class NNUNetRegressor(nn.Module):
    """Official PlainConvUNet with one unrestricted continuous output channel.

    Input: [B,8,H,W]. Output: [B,1,H,W]. There is no softmax or sigmoid.
    Deep supervision is disabled so the packaged network returns one tensor.
    """

    def __init__(self, input_channels=8, out_channels=1,
                 features_per_stage=(32, 64, 128, 256, 320),
                 n_conv_per_stage=2, n_conv_per_stage_decoder=2):
        super().__init__()
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        features = list(features_per_stage)
        if len(features) < 2 or any(v < 1 for v in features):
            raise ValueError("Use at least two stages with positive channel counts")
        if input_channels < 1 or out_channels < 1:
            raise ValueError("Channel counts must be positive")
        if n_conv_per_stage < 1 or n_conv_per_stage_decoder < 1:
            raise ValueError("Convolution counts must be positive")
        self.input_channels = input_channels
        self.divisor = 2 ** (len(features) - 1)
        self.network = get_network_from_plans(
            arch_class_name="dynamic_network_architectures.architectures.unet.PlainConvUNet",
            arch_kwargs=dict(
                n_stages=len(features), features_per_stage=features,
                conv_op="torch.nn.Conv2d", kernel_sizes=[[3, 3]] * len(features),
                strides=[[1, 1]] + [[2, 2]] * (len(features) - 1),
                n_conv_per_stage=[n_conv_per_stage] * len(features),
                n_conv_per_stage_decoder=[n_conv_per_stage_decoder] * (len(features) - 1),
                conv_bias=True, norm_op="torch.nn.InstanceNorm2d",
                norm_op_kwargs=dict(eps=1e-5, affine=True),
                dropout_op=None, dropout_op_kwargs=None,
                nonlin="torch.nn.LeakyReLU",
                nonlin_kwargs=dict(negative_slope=1e-2, inplace=True),
            ),
            arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
            input_channels=input_channels, output_channels=out_channels,
            allow_init=True, deep_supervision=False,
        )

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != self.input_channels:
            raise ValueError(f"Expected [B,{self.input_channels},H,W]")
        height, width = image.shape[-2:]
        if min(height, width) < 2 * self.divisor:
            raise ValueError(f"Grid dimensions must be >= {2 * self.divisor} for InstanceNorm")
        # Preserve the physical grid; only pad at the right/bottom if necessary.
        pad_h, pad_w = (-height) % self.divisor, (-width) % self.divisor
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
        prediction = self.network(image)
        return prediction[..., :height, :width]
