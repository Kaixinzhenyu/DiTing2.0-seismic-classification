
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset

from config import DATA_ROOT, PT_FILES, TaskSpec


class PtTensorDataset(Dataset):
    """
    A light wrapper around a single .pt file with keys:
      - x: torch.Tensor, shape (N, C, H, W)
      - y: torch.Tensor, shape (N,)
      - meta: dict (optional)
    """
    def __init__(self, pt_path: str):
        super().__init__()
        obj = torch.load(pt_path, map_location="cpu")

        # Support both dict and tuple/list formats defensively
        if isinstance(obj, dict):
            self.x = obj["x"]
            self.y = obj["y"]
            self.meta = obj.get("meta", {})
        elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
            self.x, self.y = obj[0], obj[1]
            self.meta = {}
        else:
            raise ValueError(f"Unsupported .pt format: {pt_path}")

        if not torch.is_tensor(self.x) or not torch.is_tensor(self.y):
            raise ValueError(f"x/y must be torch.Tensor in {pt_path}")

        if self.x.size(0) != self.y.size(0):
            raise ValueError(f"Length mismatch in {pt_path}: x={self.x.size(0)} vs y={self.y.size(0)}")

    def __len__(self) -> int:
        return int(self.x.size(0))

    def __getitem__(self, idx: int):
        return self.x[idx], int(self.y[idx])


def _join(root: str, rel: str) -> str:
    return os.path.join(root, rel)


def get_split_paths(task: TaskSpec) -> Dict[str, List[str]]:
    """
    Returns:
      {
        "train": [pt_path1, pt_path2, ...],
        "valid": [...],
        "test":  [...]
      }
    """
    # base 3 classes always included
    base_classes = ["Earthquake", "Explosion", "Collapse"]

    noise_key: Optional[str] = None
    if task.num_classes == 4:
        if task.noise_variant is None:
            raise ValueError("4-class task must specify noise_variant")
        noise_key = f"Noise_{task.noise_variant}"
        if noise_key not in PT_FILES:
            raise KeyError(f"Noise key not found in PT_FILES: {noise_key}")

    split_paths: Dict[str, List[str]] = {"train": [], "valid": [], "test": []}
    for split in ["train", "valid", "test"]:
        for cls in base_classes:
            split_paths[split].append(_join(DATA_ROOT, PT_FILES[cls][split]))
        if noise_key is not None:
            split_paths[split].append(_join(DATA_ROOT, PT_FILES[noise_key][split]))

    return split_paths


def build_datasets(task: TaskSpec) -> Tuple[Dataset, Dataset, Dataset, Dict[str, Dict[int, int]]]:
    """
    Build ConcatDataset for train/valid/test.

    Returns:
      train_ds, valid_ds, test_ds, stats
    stats example:
      {
        "train": {0: 43000, 1: 42000, 2: 41000, 3: 25000},
        "valid": {...},
        "test":  {...}
      }
    """
    split_paths = get_split_paths(task)

    stats: Dict[str, Dict[int, int]] = {}
    datasets: Dict[str, Dataset] = {}

    for split, paths in split_paths.items():
        parts: List[Dataset] = []
        counts: Dict[int, int] = {}
        for p in paths:
            ds = PtTensorDataset(p)
            parts.append(ds)

            y = ds.y.to(torch.long)
            bc = torch.bincount(y)
            for label_id, c in enumerate(bc.tolist()):
                if c == 0:
                    continue
                counts[label_id] = counts.get(label_id, 0) + int(c)

        datasets[split] = ConcatDataset(parts)
        stats[split] = dict(sorted(counts.items(), key=lambda kv: kv[0]))

    return datasets["train"], datasets["valid"], datasets["test"], stats
