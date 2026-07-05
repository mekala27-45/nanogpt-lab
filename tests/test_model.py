"""Model shapes, weight tying, parameter counting, and gradient sanity."""

from __future__ import annotations

import torch

from _helpers import VOCAB, make_cfg
from nanogpt_lab.config import ModelCfg
from nanogpt_lab.model.transformer import GPT


def test_forward_shapes(tiny_model: GPT, model_cfg: ModelCfg) -> None:
    idx = torch.randint(0, VOCAB, (3, model_cfg.block_size))
    logits, loss = tiny_model(idx, idx)
    assert logits.shape == (3, model_cfg.block_size, VOCAB)
    assert loss is not None and loss.ndim == 0


def test_weight_tying() -> None:
    model = GPT(make_cfg(tie_weights=True), VOCAB)
    assert model.lm_head.weight is model.tok_emb.weight


def test_learned_pos_has_more_params() -> None:
    rope = GPT(make_cfg(pos="rope"), VOCAB).num_params(non_embedding=False)
    learned = GPT(make_cfg(pos="learned"), VOCAB).num_params(non_embedding=False)
    assert learned > rope  # the learned position table adds parameters


def test_overfit_single_batch() -> None:
    """A tiny model must be able to memorize one random batch (loss -> ~0)."""
    torch.manual_seed(0)
    cfg = make_cfg()
    model = GPT(cfg, VOCAB)
    x = torch.randint(0, VOCAB, (4, cfg.block_size))
    y = torch.randint(0, VOCAB, (4, cfg.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    first = None
    last = None
    for i in range(300):
        _, loss = model(x, y)
        assert loss is not None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()

    assert first is not None and last is not None
    assert last < 0.1 * first  # memorized the batch
