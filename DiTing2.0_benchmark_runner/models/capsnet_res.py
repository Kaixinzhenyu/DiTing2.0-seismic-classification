
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .capsnet import squash, AgreementRouting, PrimaryCapsLayer, CapsLayer


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + x)


class CapsNetRes(nn.Module):
    """
    CapsNet with a small residual backbone (keeps H,W unchanged),
    then PrimaryCaps -> DigitCaps.
    """
    def __init__(
        self,
        n_classes: int = 3,
        routing_iterations: int = 3,
        input_height: int = 40,
        input_width: int = 72,
        conv_channels: int = 256,
        primary_caps: int = 16,
        primary_dim: int = 16,
        digit_dim: int = 16,
        num_res_blocks: int = 2,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.n_classes = int(n_classes)

        self.stem = nn.Sequential(
            nn.Conv2d(3, conv_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
        )

        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock(conv_channels))
        self.backbone = nn.Sequential(*blocks)

        self.primary = PrimaryCapsLayer(conv_channels, primary_caps, primary_dim, kernel_size=3, stride=2)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_height, input_width)
            out = self.stem(dummy)
            out = self.backbone(out)
            out = self.primary(out)
            num_primary_caps = int(out.size(1))

        routing = AgreementRouting(num_primary_caps, self.n_classes, routing_iterations)
        self.digit = CapsLayer(num_primary_caps, primary_dim, self.n_classes, digit_dim, routing)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.backbone(out)
        out = self.dropout(out)
        out = self.primary(out)
        out = self.dropout(out)
        out = self.digit(out)
        probs = torch.sqrt((out ** 2).sum(dim=-1) + 1e-8)
        return probs


def get_capsnet_res_model(
    n_classes: int = 3,
    routing_iterations: int = 3,
    input_height: int = 40,
    input_width: int = 72,
) -> CapsNetRes:
    return CapsNetRes(
        n_classes=n_classes,
        routing_iterations=routing_iterations,
        input_height=input_height,
        input_width=input_width,
    )
