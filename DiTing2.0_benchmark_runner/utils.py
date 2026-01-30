from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


class Timer:
    def __init__(self):
        self.t0 = None

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def stop(self) -> float:
        if self.t0 is None:
            raise RuntimeError("Timer not started")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return float(t1 - self.t0)


@torch.no_grad()
def count_flops_thop(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int],
    device: torch.device,
    *,
    batch_size: int = 1,
) -> Dict[str, Optional[float]]:
    """
    Compute MACs/FLOPs using thop.

    Returns a dict suitable for JSON:
      {
        "macs": int,
        "flops": int,          # we define flops = 2 * macs (common convention)
        "gmacs": float,
        "gflops": float,
        "method": "thop",
        "error": None or str
      }
    """
    out: Dict[str, Optional[float]] = {
        "macs": None,
        "flops": None,
        "gmacs": None,
        "gflops": None,
        "method": "thop",
        "error": None,
    }

    try:
        from thop import profile  # type: ignore
    except Exception as e:
        out["error"] = f"thop_not_installed: {e}"
        return out

    try:
        c, h, w = input_shape
        dummy = torch.randn(batch_size, c, h, w, device=device, dtype=torch.float32)

        was_training = model.training
        model.eval()

        macs, params = profile(model, inputs=(dummy,), verbose=False)

        model.train(was_training)

        macs = float(macs)
        flops = 2.0 * macs

        out["macs"] = int(macs)
        out["flops"] = int(flops)
        out["gmacs"] = float(macs / 1e9)
        out["gflops"] = float(flops / 1e9)
        return out

    except Exception as e:
        out["error"] = f"thop_failed: {repr(e)}"
        return out
