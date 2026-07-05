"""RoPE properties: it's a rotation (norm-preserving) and encodes relative position."""

from __future__ import annotations

import pytest
import torch

from nanogpt_lab.model.rope import apply_rope, build_rope_cache


def test_rope_preserves_norm() -> None:
    cos, sin = build_rope_cache(seq_len=8, head_dim=16, theta=10000.0)
    x = torch.randn(1, 1, 8, 16)
    y = apply_rope(x, cos[:8], sin[:8])
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope_is_relative() -> None:
    """(R_m q)·(R_n k) depends only on the offset m-n, not absolute m, n."""
    cos, sin = build_rope_cache(seq_len=64, head_dim=16, theta=10000.0)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def dot(m: int, n: int) -> torch.Tensor:
        qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum()

    assert torch.allclose(dot(5, 3), dot(20, 18), atol=1e-4)  # both offset +2
    assert torch.allclose(dot(10, 15), dot(2, 7), atol=1e-4)  # both offset -5


def test_rope_requires_even_head_dim() -> None:
    with pytest.raises(ValueError, match="even"):
        build_rope_cache(seq_len=4, head_dim=7, theta=10000.0)
