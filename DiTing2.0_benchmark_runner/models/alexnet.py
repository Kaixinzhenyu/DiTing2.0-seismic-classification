
from __future__ import annotations

import torch
import torch.nn as nn


def _dummy_input(dataset_choose: str) -> torch.Tensor:
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


class AlexNet(nn.Module):
    def __init__(self, dataset_choose: str, num_classes: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        with torch.no_grad():
            dummy = _dummy_input(dataset_choose)
            out = self.features(dummy)
            flat = int(out.view(out.size(0), -1).size(1))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(flat, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def alexnet(dataset_choose: str, num_classes: int = 3) -> AlexNet:
    return AlexNet(dataset_choose=dataset_choose, num_classes=num_classes)
