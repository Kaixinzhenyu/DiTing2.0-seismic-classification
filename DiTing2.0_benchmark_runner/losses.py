
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):
    """
    Capsule Network margin loss.

    L_k = T_k * max(0, m_plus - v_k)^2 + lambda*(1-T_k)*max(0, v_k - m_minus)^2
    where v_k is the length (prob) of capsule k.

    This implementation expects:
      - inputs: shape (B, num_classes), non-negative (capsule lengths)
      - targets: shape (B,), int64 class indices
    """
    def __init__(self, m_plus: float = 0.9, m_minus: float = 0.1, lambda_: float = 0.5):
        super().__init__()
        self.m_plus = float(m_plus)
        self.m_minus = float(m_minus)
        self.lambda_ = float(lambda_)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError(f"MarginLoss expects inputs (B,C), got {inputs.shape}")
        b, c = inputs.shape
        if targets.ndim != 1:
            targets = targets.view(-1)
        if targets.size(0) != b:
            raise ValueError(f"Targets length mismatch: {targets.size(0)} vs {b}")

        # One-hot
        T = F.one_hot(targets, num_classes=c).float()

        v = inputs
        loss_pos = T * torch.clamp(self.m_plus - v, min=0.0) ** 2
        loss_neg = (1.0 - T) * torch.clamp(v - self.m_minus, min=0.0) ** 2
        loss = loss_pos + self.lambda_ * loss_neg
        return loss.sum(dim=1).mean()
