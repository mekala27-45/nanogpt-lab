"""Shared fixtures: a tiny model config and model for fast, offline tests."""

from __future__ import annotations

import pytest
import torch

from _helpers import VOCAB, make_cfg
from nanogpt_lab.config import ModelCfg
from nanogpt_lab.model.transformer import GPT


@pytest.fixture
def model_cfg() -> ModelCfg:
    return make_cfg()


@pytest.fixture
def tiny_model(model_cfg: ModelCfg) -> GPT:
    torch.manual_seed(0)
    return GPT(model_cfg, vocab_size=VOCAB).eval()
