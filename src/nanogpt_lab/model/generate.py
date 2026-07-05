"""Autoregressive sampling + checkpoint loading for the generation CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from nanogpt_lab.config import PROJECT_ROOT, Config, load_config, resolve_device
from nanogpt_lab.data.tokenizer import CharTokenizer
from nanogpt_lab.model.transformer import GPT


@torch.no_grad()
def generate(
    model: GPT,
    idx: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``max_new_tokens`` tokens, cropping context to block_size each step.

    This path is correct for any output length (unlike a fixed KV cache) and is
    what the demo uses; the cached path is validated separately in the tests.
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k).values[:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat((idx, next_id), dim=1)
    return idx


@torch.no_grad()
def incremental_logits(model: GPT, idx: Tensor) -> Tensor:
    """Per-position logits via one-token-at-a-time KV-cache decoding (B, T, V)."""
    model.eval()
    caches = model.new_caches()
    steps = []
    for t in range(idx.size(1)):
        logits, _ = model(idx[:, t : t + 1], caches=caches, start_pos=t)
        steps.append(logits[:, -1, :])
    return torch.stack(steps, dim=1)


def load_checkpoint(path: Path, device: str) -> tuple[GPT, CharTokenizer, Config]:
    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    config = Config.model_validate(ckpt["config"])
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])
    model = GPT(config.model, tokenizer.vocab_size)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, tokenizer, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--config", default="config/smoke.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.train.device)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else PROJECT_ROOT / config.train.out_dir / "best.pt"
    )
    model, tokenizer, _ = load_checkpoint(ckpt_path, device)

    prompt = args.prompt if args.prompt is not None else config.generate.prompt
    max_new = args.max_new_tokens or config.generate.max_new_tokens
    ids = torch.tensor([tokenizer.encode(prompt) or [0]], dtype=torch.long, device=device)
    out = generate(model, ids, max_new, config.generate.temperature, config.generate.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
