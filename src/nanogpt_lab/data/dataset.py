"""Download a text corpus, build a tokenizer, and split into train/val tensors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import requests
import torch
from torch import Tensor

from nanogpt_lab.config import PROJECT_ROOT, DataCfg
from nanogpt_lab.data.tokenizer import CharTokenizer
from nanogpt_lab.logging_setup import get_logger

log = get_logger(__name__)
_TIMEOUT_S = 120


@dataclass
class Dataset:
    train_ids: Tensor
    val_ids: Tensor
    tokenizer: CharTokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def download_text(cfg: DataCfg) -> str:
    """Fetch (and cache) the corpus. Verifies SHA-256 when one is configured."""
    cache_dir = _resolve(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{cfg.corpus}.txt"

    if dest.exists():
        data = dest.read_bytes()
        if not cfg.sha256 or hashlib.sha256(data).hexdigest() == cfg.sha256:
            log.info("data.cache_hit", path=str(dest), bytes=len(data))
            return data.decode("utf-8")

    log.info("data.download", url=cfg.url)
    resp = requests.get(cfg.url, timeout=_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.content
    if cfg.sha256:
        actual = hashlib.sha256(data).hexdigest()
        if actual != cfg.sha256:
            raise ValueError(f"Checksum mismatch: expected {cfg.sha256}, got {actual}")
    dest.write_bytes(data)
    log.info("data.downloaded", path=str(dest), bytes=len(data))
    return data.decode("utf-8")


def load_dataset(cfg: DataCfg) -> Dataset:
    text = download_text(cfg)
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_train = int(len(ids) * (1.0 - cfg.val_fraction))
    dataset = Dataset(train_ids=ids[:n_train], val_ids=ids[n_train:], tokenizer=tokenizer)
    log.info(
        "data.loaded",
        vocab=tokenizer.vocab_size,
        train_tokens=int(dataset.train_ids.numel()),
        val_tokens=int(dataset.val_ids.numel()),
    )
    return dataset
