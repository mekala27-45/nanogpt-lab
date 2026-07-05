"""Rotary Positional Embeddings (RoPE), from scratch.

RoPE encodes *absolute* position as a rotation applied to query/key vectors,
such that the attention dot product depends only on the *relative* offset
between two positions. We use the GPT-NeoX / Llama "rotate-half" convention.

References: Su et al., "RoFormer" (2021); the Llama implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    """Precompute (cos, sin) tables of shape (seq_len, head_dim).

    ``head_dim`` must be even. Frequencies follow the standard geometric schedule
    ``theta ** (-2i/d)``; each frequency is duplicated so the table is full width.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x`` (B, n_head, T, head_dim) by the (T, head_dim) cos/sin slice."""
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + _rotate_half(x) * sin
