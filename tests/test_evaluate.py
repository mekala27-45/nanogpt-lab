"""Batching correctness, the LR schedule, and metric conversions."""

from __future__ import annotations

import math

import torch

from nanogpt_lab.evaluate import bits_per_char, get_batch, perplexity
from nanogpt_lab.train import cosine_lr


def test_get_batch_targets_are_shifted_inputs() -> None:
    data = torch.arange(100)
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(data, block_size=8, batch_size=4, device="cpu", generator=g)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    # data is a ramp, so each target is exactly input + 1.
    assert torch.equal(y, x + 1)


def test_cosine_lr_schedule() -> None:
    warmup, max_iters, lr, min_lr = 10, 100, 1e-3, 1e-4
    assert cosine_lr(0, warmup, max_iters, lr, min_lr) < lr  # warming up
    assert math.isclose(cosine_lr(warmup, warmup, max_iters, lr, min_lr), lr, rel_tol=1e-6)
    assert math.isclose(cosine_lr(max_iters, warmup, max_iters, lr, min_lr), min_lr)
    # monotonic decay through the cosine phase
    mid_early = cosine_lr(30, warmup, max_iters, lr, min_lr)
    mid_late = cosine_lr(70, warmup, max_iters, lr, min_lr)
    assert lr > mid_early > mid_late > min_lr


def test_metric_conversions() -> None:
    assert math.isclose(bits_per_char(math.log(2)), 1.0)
    assert math.isclose(perplexity(0.0), 1.0)
    assert math.isclose(perplexity(math.log(5)), 5.0, rel_tol=1e-6)
