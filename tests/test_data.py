"""Dataset download/verify/split — offline via monkeypatched HTTP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nanogpt_lab.config import DataCfg
from nanogpt_lab.data import dataset


class _Resp:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def _cfg(tmp_path: Path, sha: str = "") -> DataCfg:
    return DataCfg(
        corpus="t", url="http://example/x", sha256=sha, cache_dir=str(tmp_path), val_fraction=0.1
    )


def test_checksum_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset.requests, "get", lambda *a, **k: _Resp(b"hello"))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        dataset.download_text(_cfg(tmp_path, sha="deadbeef"))


def test_download_without_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset.requests, "get", lambda *a, **k: _Resp(b"hello"))
    assert dataset.download_text(_cfg(tmp_path)) == "hello"


def test_load_dataset_splits_and_builds_vocab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(cfg: Any) -> str:
        return "abcde" * 200  # 1000 chars, vocab of 5

    monkeypatch.setattr(dataset, "download_text", fake_download)
    ds = dataset.load_dataset(_cfg(tmp_path))
    assert ds.vocab_size == 5
    assert ds.train_ids.numel() + ds.val_ids.numel() == 1000
    assert ds.val_ids.numel() == 100
