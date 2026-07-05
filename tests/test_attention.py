"""Causality: a change to a later token must not affect earlier-position outputs."""

from __future__ import annotations

import torch

from _helpers import VOCAB
from nanogpt_lab.config import ModelCfg
from nanogpt_lab.model.transformer import GPT


def test_model_is_causal(model_cfg: ModelCfg) -> None:
    torch.manual_seed(0)
    model = GPT(model_cfg, vocab_size=VOCAB).eval()
    idx = torch.randint(0, VOCAB, (1, model_cfg.block_size))

    with torch.no_grad():
        base, _ = model(idx)
        idx2 = idx.clone()
        idx2[0, -1] = (idx2[0, -1] + 1) % VOCAB  # perturb only the last token
        alt, _ = model(idx2)

    # Every position except the last must be untouched.
    assert torch.allclose(base[:, :-1, :], alt[:, :-1, :], atol=1e-5)
    assert not torch.allclose(base[:, -1, :], alt[:, -1, :])
