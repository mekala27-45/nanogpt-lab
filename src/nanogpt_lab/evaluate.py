"""Batching and loss/perplexity estimation."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from nanogpt_lab.model.transformer import GPT


def get_batch(
    data: Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample a random (inputs, next-token-targets) minibatch from a token stream."""
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: Tensor,
    val_data: Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Mean cross-entropy over ``eval_iters`` batches for train and val splits."""
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device, generator)
            _, loss = model(x, y)
            assert loss is not None
            losses[k] = loss.item()
        out[split] = float(losses.mean())
    if was_training:
        model.train()
    return out


def bits_per_char(loss: float) -> float:
    """Cross-entropy (nats) → bits per character."""
    return loss / math.log(2)


def perplexity(loss: float) -> float:
    return math.exp(loss)
