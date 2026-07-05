"""Normalization and MLP variants — the pieces the ablation study toggles."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean-centering, no bias) — as in Llama."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "rms":
        return RMSNorm(dim)
    if kind == "layer":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm kind: {kind!r}")


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: (SiLU(xW1) * xW3) W2 — the Llama/PaLM MLP."""

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GELUMLP(nn.Module):
    """Classic GPT-2 feed-forward: GELU(xW1) W2."""

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, hidden)
        self.proj = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


def make_mlp(kind: str, dim: int, ratio: int, dropout: float) -> nn.Module:
    hidden = ratio * dim
    if kind == "swiglu":
        return SwiGLU(dim, hidden, dropout)
    if kind == "gelu":
        return GELUMLP(dim, hidden, dropout)
    raise ValueError(f"unknown mlp kind: {kind!r}")
