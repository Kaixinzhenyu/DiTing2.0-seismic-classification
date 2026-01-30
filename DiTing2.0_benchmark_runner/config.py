"""
Dynamic configuration area (edit this file first).

This project is designed to reproduce/extend your GJI revision experiments:

- GPU0: 3-class (Earthquake / Explosion / Collapse) without noise
- GPU1: 4-class (Earthquake / Explosion / Collapse / Noise) with 4 noise subsets:
        Expert_A, Expert_B, Expert_C, Junior_A

Each (model, task) runs 5 seeds, and the script outputs mean/std for:
Accuracy, Macro-F1, Weighted-F1, plus training time, inference time, params, confusion matrix, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------
# 1) PATHS
# ---------------------------
DATA_ROOT: str = "/home/zypei/DiTing2.0_dataset/remake_noise_experiment/EQ_EP_SS_Noise_datasets_enhancement_MFCC"
OUTPUT_ROOT: str = "./runs_gji_revision"


# ---------------------------
# 2) DATASET FILE MAP (relative to DATA_ROOT)
# ---------------------------
PT_FILES: Dict[str, Dict[str, str]] = {
    "Earthquake": {
        "train": "Earthquake_train_dataset_enhancement_MFCC.pt",
        "valid": "Earthquake_valid_dataset_enhancement_MFCC.pt",
        "test":  "Earthquake_test_dataset_enhancement_MFCC.pt",
    },
    "Explosion": {
        "train": "Explosion_train_dataset_enhancement_MFCC.pt",
        "valid": "Explosion_valid_dataset_enhancement_MFCC.pt",
        "test":  "Explosion_test_dataset_enhancement_MFCC.pt",
    },
    "Collapse": {
        "train": "Collapse_train_dataset_enhancement_MFCC.pt",
        "valid": "Collapse_valid_dataset_enhancement_MFCC.pt",
        "test":  "Collapse_test_dataset_enhancement_MFCC.pt",
    },
    "Noise_Expert_A": {
        "train": "Noise_Expert_A_train_dataset_MFCC.pt",
        "valid": "Noise_Expert_A_valid_dataset_MFCC.pt",
        "test":  "Noise_Expert_A_test_dataset_MFCC.pt",
    },
    "Noise_Expert_B": {
        "train": "Noise_Expert_B_train_dataset_MFCC.pt",
        "valid": "Noise_Expert_B_valid_dataset_MFCC.pt",
        "test":  "Noise_Expert_B_test_dataset_MFCC.pt",
    },
    "Noise_Expert_C": {
        "train": "Noise_Expert_C_train_dataset_MFCC.pt",
        "valid": "Noise_Expert_C_valid_dataset_MFCC.pt",
        "test":  "Noise_Expert_C_test_dataset_MFCC.pt",
    },
    "Noise_Junior_A": {
        "train": "Noise_Junior_A_train_dataset_MFCC.pt",
        "valid": "Noise_Junior_A_valid_dataset_MFCC.pt",
        "test":  "Noise_Junior_A_test_dataset_MFCC.pt",
    },
}

# 3-class = no noise; 4-class = with noise subset
NOISE_VARIANTS: List[str] = ["Expert_B", "Expert_C", "Junior_A"]


# ---------------------------
# 3) LABELS
# ---------------------------
CLASS_NAMES_3: List[str] = ["Earthquake", "Explosion", "Collapse"]
CLASS_NAMES_4: List[str] = ["Earthquake", "Explosion", "Collapse", "Noise"]


# ---------------------------
# 4) INPUT SHAPE (MFCC: 3 x 40 x 72)
# ---------------------------
IN_CHANNELS: int = 3
INPUT_HEIGHT: int = 40
INPUT_WIDTH: int = 72
DATASET_CHOOSE_FOR_2D: str = "MFCC_40_72"


# ---------------------------
# 5) MODELS TO RUN
# ---------------------------
# ⚠️ 你原来写成 "vgg""alexnet"... 会变成字符串拼接，直接炸
# ⚠️ 这里的 key 必须和 models/__init__.py 一致（你仓库里常见的是下面这些）
MODELS_TO_RUN: List[str] = [
    "alexnet",
    "vgg11_bn",
    "resnet18",
    "googlenet",
    "vit",
    "capsnet",
    "capsnet_res",
]


# ---------------------------
# 6) TRAINING HYPERPARAMETERS
# ---------------------------
# ✅ 只跑 1 个 seed：全局改成 [0]
SEEDS: List[int] = [0]

EPOCHS: int = 100
BATCH_SIZE: int = 256
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-4

OPTIMIZER: str = "adam"

NUM_WORKERS: int = 8
PIN_MEMORY: bool = True

DETERMINISTIC: bool = False
USE_AMP: bool = True

BEST_METRIC: str = "macro_f1"  # "macro_f1" or "acc"


# ---------------------------
# 6.1) EARLY STOPPING (NEW)
# ---------------------------
EARLY_STOP: bool = True
EARLY_STOP_PATIENCE: int = 10       # 连续多少个 epoch 无提升就停
EARLY_STOP_MIN_DELTA: float = 1e-4  # 认为“提升”的最小增量
EARLY_STOP_MIN_EPOCHS: int = 10     # 至少跑满多少 epoch 才允许早停


# ---------------------------
# 7) MODEL-SPECIFIC SETTINGS
# ---------------------------
VIT_CFG: Dict[str, int] = dict(
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=512,
    patch_height=5,
    patch_width=9,
)

CAPSNET_CFG: Dict[str, float] = dict(
    routing_iterations=3,
    m_plus=0.9,
    m_minus=0.1,
    lambda_=0.5,
)


# ---------------------------
# 8) OPTIONAL: LOSS WEIGHTING
# ---------------------------
USE_CLASS_WEIGHTS: bool = False


@dataclass
class TaskSpec:
    """
    A single experiment task (dataset definition).
    """
    name: str
    num_classes: int
    class_names: List[str]
    noise_variant: Optional[str] = None  # None for 3-class; for 4-class use one of NOISE_VARIANTS


def build_tasks(mode: str) -> List[TaskSpec]:
    """
    mode:
      - "3class": one task
      - "4class": tasks per NOISE_VARIANTS
    """
    mode = mode.lower().strip()
    if mode == "3class":
        return [TaskSpec(name="3class_no_noise", num_classes=3, class_names=CLASS_NAMES_3, noise_variant=None)]
    if mode == "4class":
        tasks: List[TaskSpec] = []
        for v in NOISE_VARIANTS:
            tasks.append(TaskSpec(name=f"4class_noise_{v}", num_classes=4, class_names=CLASS_NAMES_4, noise_variant=v))
        return tasks
    raise ValueError(f"Unknown mode: {mode}. Use '3class' or '4class'.")
