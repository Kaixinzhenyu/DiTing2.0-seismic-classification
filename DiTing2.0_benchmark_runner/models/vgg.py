
from __future__ import annotations

import torch
import torch.nn as nn


cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # VGG11
}


def _dummy_input(dataset_choose: str) -> torch.Tensor:
    # Extend if you add other input shapes later
    if dataset_choose in ("MFCC_40_72", "mfcc"):
        return torch.randn(1, 3, 40, 72)
    if dataset_choose in ("STFT_72_72", "stft"):
        return torch.randn(1, 3, 72, 72)
    if dataset_choose == "GADF":
        return torch.randn(1, 3, 128, 128)
    if dataset_choose == "MFCC_36_72":
        return torch.randn(1, 3, 36, 72)
    if dataset_choose == "MFCC":
        return torch.randn(1, 3, 39, 24)
    if dataset_choose == "STFT":
        return torch.randn(1, 3, 26, 118)
    raise ValueError(f"Unknown dataset_choose: {dataset_choose}")


class VGG(nn.Module):
    def __init__(self, features: nn.Module, dataset_choose: str, num_class: int = 3):
        super().__init__()
        self.features = features

        with torch.no_grad():
            dummy = _dummy_input(dataset_choose)
            dummy_out = self.features(dummy)
            out_features = int(dummy_out.view(dummy_out.size(0), -1).size(1))

        self.classifier = nn.Sequential(
            nn.Linear(out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


def make_layers(cfg_list, batch_norm: bool = False) -> nn.Sequential:
    layers = []
    in_ch = 3
    for v in cfg_list:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        layers.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(v))
        layers.append(nn.ReLU(inplace=True))
        in_ch = v
    return nn.Sequential(*layers)


def vgg11_bn(dataset_choose: str, num_classes: int = 3) -> VGG:
    return VGG(make_layers(cfg["A"], batch_norm=True), dataset_choose=dataset_choose, num_class=num_classes)
