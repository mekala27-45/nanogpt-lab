"""Sampling: output shape, seeded determinism, and greedy determinism."""

from __future__ import annotations

import torch

from nanogpt_lab.model.generate import generate
from nanogpt_lab.model.transformer import GPT


def test_generate_shape(tiny_model: GPT) -> None:
    idx = torch.zeros(1, 1, dtype=torch.long)
    out = generate(tiny_model, idx, max_new_tokens=20, temperature=1.0, top_k=5)
    assert out.shape == (1, 21)  # grows beyond block_size via context cropping


def test_generate_seeded_determinism(tiny_model: GPT) -> None:
    idx = torch.zeros(1, 1, dtype=torch.long)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    o1 = generate(tiny_model, idx, 15, temperature=1.0, top_k=5, generator=g1)
    o2 = generate(tiny_model, idx, 15, temperature=1.0, top_k=5, generator=g2)
    assert torch.equal(o1, o2)


def test_generate_greedy_is_deterministic(tiny_model: GPT) -> None:
    idx = torch.zeros(1, 1, dtype=torch.long)
    o1 = generate(tiny_model, idx, 15, temperature=1e-6, top_k=1)
    o2 = generate(tiny_model, idx, 15, temperature=1e-6, top_k=1)
    assert torch.equal(o1, o2)
