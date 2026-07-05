"""The headline correctness test: cached incremental decoding == full forward.

If RoPE positions or the cache mask were wrong, these would diverge. We check it
across every architecture variant the ablation can produce.
"""

from __future__ import annotations

import pytest
import torch

from _helpers import VOCAB, make_cfg
from nanogpt_lab.config import ModelCfg
from nanogpt_lab.model.generate import incremental_logits
from nanogpt_lab.model.transformer import GPT

_VARIANTS = [
    {},  # baseline: rope + rms + swiglu
    {"pos": "learned"},
    {"norm": "layer"},
    {"mlp": "gelu"},
    {"pos": "learned", "norm": "layer", "mlp": "gelu"},  # classic GPT-2 style
]


@pytest.mark.parametrize("override", _VARIANTS)
def test_incremental_matches_full_forward(override: dict[str, str]) -> None:
    cfg: ModelCfg = make_cfg(**override)
    torch.manual_seed(0)
    model = GPT(cfg, vocab_size=VOCAB).eval()
    idx = torch.randint(0, VOCAB, (2, cfg.block_size))

    with torch.no_grad():
        full, _ = model(idx)
        incremental = incremental_logits(model, idx)

    assert torch.allclose(
        full, incremental, atol=1e-4
    ), f"KV-cache decoding diverged from full forward for variant {override}"
