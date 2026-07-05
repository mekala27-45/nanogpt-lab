"""Training loop: AdamW + cosine LR with warmup, grad clipping, checkpointing.

Run: ``python -m nanogpt_lab.train --config config/smoke.yaml``. Produces a
best-val checkpoint, a training-history JSON, and a generated text sample.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from nanogpt_lab.config import PROJECT_ROOT, Config, get_settings, load_config, resolve_device
from nanogpt_lab.data.dataset import load_dataset
from nanogpt_lab.data.tokenizer import CharTokenizer
from nanogpt_lab.evaluate import bits_per_char, estimate_loss, get_batch, perplexity
from nanogpt_lab.logging_setup import configure_logging, get_logger
from nanogpt_lab.model.generate import generate
from nanogpt_lab.model.transformer import GPT

log = get_logger(__name__)


@dataclass
class TrainResult:
    best_val_loss: float
    val_bpc: float
    val_perplexity: float
    n_params: int
    device: str
    sample: str
    checkpoint: Path | None
    history: list[dict[str, float]] = field(default_factory=list)


def cosine_lr(it: int, warmup: int, max_iters: int, lr: float, min_lr: float) -> float:
    if it < warmup:
        return lr * (it + 1) / (warmup + 1)
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup) / max(max_iters - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def configure_optimizer(model: GPT, weight_decay: float, lr: float) -> torch.optim.Optimizer:
    """Weight-decay 2D params (matmuls/embeddings); leave norms/biases undecayed."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def _save_checkpoint(
    path: Path, model: GPT, config: Config, tokenizer: CharTokenizer, it: int, val_loss: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": config.model_dump(),
        "model": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "iter": it,
        "val_loss": val_loss,
    }
    torch.save(payload, path)


def train(config: Config) -> TrainResult:
    torch.manual_seed(config.seed)
    device = resolve_device(config.train.device)
    tcfg = config.train

    dataset = load_dataset(config.data)
    model = GPT(config.model, dataset.vocab_size).to(device)
    n_params = model.num_params()
    log.info("model.built", n_params=n_params, vocab=dataset.vocab_size, device=device)

    optimizer = configure_optimizer(model, tcfg.weight_decay, tcfg.lr)
    batch_gen = torch.Generator().manual_seed(config.seed)

    out_dir = PROJECT_ROOT / tcfg.out_dir
    ckpt_path = out_dir / "best.pt"
    best_val = float("inf")
    history: list[dict[str, float]] = []

    model.train()
    for it in range(tcfg.max_iters + 1):
        lr = cosine_lr(it, tcfg.warmup_iters, tcfg.max_iters, tcfg.lr, tcfg.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % tcfg.eval_interval == 0 or it == tcfg.max_iters:
            losses = estimate_loss(
                model,
                dataset.train_ids,
                dataset.val_ids,
                config.model.block_size,
                tcfg.batch_size,
                tcfg.eval_iters,
                device,
                batch_gen,
            )
            history.append({"iter": it, "train": losses["train"], "val": losses["val"], "lr": lr})
            log.info(
                "train.eval",
                iter=it,
                train_loss=round(losses["train"], 4),
                val_loss=round(losses["val"], 4),
                val_bpc=round(bits_per_char(losses["val"]), 4),
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                _save_checkpoint(ckpt_path, model, config, dataset.tokenizer, it, best_val)

        if it == tcfg.max_iters:
            break

        x, y = get_batch(
            dataset.train_ids, config.model.block_size, tcfg.batch_size, device, batch_gen
        )
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

    # Sample from the best model for a qualitative signal.
    ids = torch.tensor(
        [dataset.tokenizer.encode(config.generate.prompt) or [0]], dtype=torch.long, device=device
    )
    out = generate(
        model,
        ids,
        config.generate.max_new_tokens,
        config.generate.temperature,
        config.generate.top_k,
    )
    sample = dataset.tokenizer.decode(out[0].tolist())

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (PROJECT_ROOT / "samples").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "samples" / "sample.txt").write_text(sample, encoding="utf-8")

    return TrainResult(
        best_val_loss=best_val,
        val_bpc=bits_per_char(best_val),
        val_perplexity=perplexity(best_val),
        n_params=n_params,
        device=device,
        sample=sample,
        checkpoint=ckpt_path if ckpt_path.exists() else None,
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NanoGPT-Lab.")
    parser.add_argument("--config", default="config/smoke.yaml")
    parser.add_argument("--max-iters", type=int, default=None, help="override train.max_iters")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    config = load_config(args.config)
    if args.max_iters is not None:
        config = config.model_copy(
            update={"train": config.train.model_copy(update={"max_iters": args.max_iters})}
        )
    result = train(config)
    log.info(
        "train.done",
        best_val_loss=round(result.best_val_loss, 4),
        val_bpc=round(result.val_bpc, 4),
        val_perplexity=round(result.val_perplexity, 2),
        n_params=result.n_params,
    )
    print("\n----- sample -----\n" + result.sample + "\n------------------")


if __name__ == "__main__":
    main()
