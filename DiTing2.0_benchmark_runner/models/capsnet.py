
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Squash function used in Capsule Networks.
    """
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    x_norm = torch.sqrt(squared_norm + eps)
    return scale * x / x_norm


class AgreementRouting(nn.Module):
    """
    Dynamic routing by agreement.
    """
    def __init__(self, input_caps: int, output_caps: int, n_iterations: int):
        super().__init__()
        self.n_iterations = int(n_iterations)
        # Routing logits b_ij
        self.b = nn.Parameter(torch.zeros(input_caps, output_caps))

    def forward(self, u_predict: torch.Tensor) -> torch.Tensor:
        """
        u_predict: (B, input_caps, output_caps, output_dim)
        returns:   (B, output_caps, output_dim)
        """
        B, input_caps, output_caps, output_dim = u_predict.size()

        # c_ij = softmax(b_ij) along output_caps
        c = F.softmax(self.b, dim=1)  # (input_caps, output_caps)
        c = c.unsqueeze(0).unsqueeze(-1)  # (1, input_caps, output_caps, 1)
        s = (c * u_predict).sum(dim=1)  # (B, output_caps, output_dim)
        v = squash(s, dim=-1)

        if self.n_iterations <= 0:
            return v

        b_batch = self.b.unsqueeze(0).expand(B, input_caps, output_caps)  # (B, input_caps, output_caps)

        for _ in range(self.n_iterations):
            v_unsq = v.unsqueeze(1)  # (B, 1, output_caps, output_dim)
            # agreement
            b_batch = b_batch + (u_predict * v_unsq).sum(dim=-1)  # (B, input_caps, output_caps)

            c = F.softmax(b_batch, dim=2).unsqueeze(-1)  # (B, input_caps, output_caps, 1)
            s = (c * u_predict).sum(dim=1)  # (B, output_caps, output_dim)
            v = squash(s, dim=-1)

        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels: int, output_caps: int, output_dim: int, kernel_size: int, stride: int):
        super().__init__()
        self.output_caps = int(output_caps)
        self.output_dim = int(output_dim)
        self.conv = nn.Conv2d(
            input_channels,
            self.output_caps * self.output_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, C, H, W)
        Output: (B, num_capsules, output_dim)
        """
        out = self.conv(x)  # (B, output_caps*output_dim, H2, W2)
        B, C, H, W = out.size()
        out = out.view(B, self.output_caps, self.output_dim, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()  # (B, output_caps, H, W, output_dim)
        out = out.view(B, -1, self.output_dim)         # (B, output_caps*H*W, output_dim)
        out = squash(out, dim=-1)
        return out


class CapsLayer(nn.Module):
    """
    DigitCaps layer.
    """
    def __init__(
        self,
        input_caps: int,
        input_dim: int,
        output_caps: int,
        output_dim: int,
        routing_module: AgreementRouting,
    ):
        super().__init__()
        self.input_caps = int(input_caps)
        self.input_dim = int(input_dim)
        self.output_caps = int(output_caps)
        self.output_dim = int(output_dim)

        # weights: (input_caps, input_dim, output_caps*output_dim)
        self.weights = nn.Parameter(torch.empty(self.input_caps, self.input_dim, self.output_caps * self.output_dim))
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.input_caps)
        with torch.no_grad():
            self.weights.uniform_(-stdv, stdv)

    def forward(self, caps_output: torch.Tensor) -> torch.Tensor:
        """
        caps_output: (B, input_caps, input_dim)
        return:      (B, output_caps, output_dim)
        """
        # (B, input_caps, 1, input_dim) x (input_caps, input_dim, output_caps*output_dim)
        # Use einsum for clarity and speed
        # u_predict: (B, input_caps, output_caps*output_dim)
        u_predict = torch.einsum("bid,ido->bio", caps_output, self.weights)
        u_predict = u_predict.view(-1, self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        return v


class CapsNet(nn.Module):
    """
    Plain CapsNet: Conv -> PrimaryCaps -> DigitCaps -> lengths.
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
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.n_classes = int(n_classes)

        # IMPORTANT: padding=1 keeps H,W unchanged for MFCC 40x72 => primary caps count = 10640
        self.conv2d = nn.Conv2d(3, conv_channels, kernel_size=3, stride=1, padding=1)
        self.primary = PrimaryCapsLayer(conv_channels, primary_caps, primary_dim, kernel_size=3, stride=2)

        # compute num_primary_caps dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_height, input_width)
            out = F.relu(self.conv2d(dummy))
            out = self.primary(out)
            num_primary_caps = int(out.size(1))

        routing = AgreementRouting(num_primary_caps, self.n_classes, routing_iterations)
        self.digit = CapsLayer(num_primary_caps, primary_dim, self.n_classes, digit_dim, routing)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv2d(x))
        out = self.primary(out)
        out = self.dropout(out)
        out = self.digit(out)  # (B, n_classes, digit_dim)
        # length as class "probability"
        probs = torch.sqrt((out ** 2).sum(dim=-1) + 1e-8)  # (B, n_classes)
        return probs


def get_capsnet_model(
    n_classes: int = 3,
    routing_iterations: int = 3,
    input_height: int = 40,
    input_width: int = 72,
) -> CapsNet:
    return CapsNet(
        n_classes=n_classes,
        routing_iterations=routing_iterations,
        input_height=input_height,
        input_width=input_width,
    )
