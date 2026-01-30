
from __future__ import annotations

from typing import Dict

from config import (
    DATASET_CHOOSE_FOR_2D,
    IN_CHANNELS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    VIT_CFG,
    CAPSNET_CFG,
)

from .alexnet import alexnet
from .vgg import vgg11_bn
from .resnet import resnet18
from .googlenet import googlenet
from .vit import vit
from .capsnet import get_capsnet_model
from .capsnet_res import get_capsnet_res_model


def build_model(model_name: str, num_classes: int):
    """
    Factory for all models used in the benchmark.

    model_name keys:
      - alexnet
      - vgg11_bn
      - resnet18
      - googlenet
      - vit
      - capsnet
      - capsnet_res
    """
    name = model_name.lower().strip()

    if name == "alexnet":
        return alexnet(dataset_choose=DATASET_CHOOSE_FOR_2D, num_classes=num_classes)

    if name == "vgg11_bn":
        return vgg11_bn(dataset_choose=DATASET_CHOOSE_FOR_2D, num_classes=num_classes)

    if name == "resnet18":
        return resnet18(num_classes=num_classes)

    if name == "googlenet":
        return googlenet(num_classes=num_classes)

    if name == "vit":
        return vit(
            image_height=INPUT_HEIGHT,
            image_width=INPUT_WIDTH,
            patch_height=VIT_CFG["patch_height"],
            patch_width=VIT_CFG["patch_width"],
            num_classes=num_classes,
            dim=VIT_CFG["dim"],
            depth=VIT_CFG["depth"],
            heads=VIT_CFG["heads"],
            mlp_dim=VIT_CFG["mlp_dim"],
            pool="cls",
            channels=IN_CHANNELS,
            dropout=0.0,
            emb_dropout=0.0,
        )

    if name == "capsnet":
        return get_capsnet_model(
            n_classes=num_classes,
            routing_iterations=int(CAPSNET_CFG["routing_iterations"]),
            input_height=INPUT_HEIGHT,
            input_width=INPUT_WIDTH,
        )

    if name == "capsnet_res":
        return get_capsnet_res_model(
            n_classes=num_classes,
            routing_iterations=int(CAPSNET_CFG["routing_iterations"]),
            input_height=INPUT_HEIGHT,
            input_width=INPUT_WIDTH,
        )

    raise KeyError(f"Unknown model_name: {model_name}")
