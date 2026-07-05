"""Test helpers shared across modules (tiny model config factory)."""

from __future__ import annotations

from nanogpt_lab.config import ModelCfg

VOCAB = 27


def make_cfg(**overrides: object) -> ModelCfg:
    base: dict[str, object] = {
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 32,
        "block_size": 16,
        "dropout": 0.0,
        "mlp_ratio": 4,
        "rope_theta": 10000.0,
        "tie_weights": True,
        "norm": "rms",
        "pos": "rope",
        "mlp": "swiglu",
    }
    base.update(overrides)
    return ModelCfg(**base)  # type: ignore[arg-type]
