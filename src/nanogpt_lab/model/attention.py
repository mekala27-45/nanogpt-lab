"""Causal multi-head self-attention with optional RoPE and a KV cache.

Two paths share one implementation:
  * training / full forward — no cache, ``is_causal=True`` (fused/flash SDPA).
  * incremental decoding — a per-layer KV cache; an explicit boolean mask lets
    the T new queries attend to all cached keys up to their own position.

The KV-cache path is proven equivalent to the full forward by
``tests/test_kv_cache.py`` — the single most important correctness check here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanogpt_lab.config import ModelCfg
from nanogpt_lab.model.rope import apply_rope


class KVCache:
    """Holds concatenated keys/values for one attention layer during decoding."""

    def __init__(self) -> None:
        self.k: Tensor | None = None
        self.v: Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else int(self.k.size(2))

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        self.k = k if self.k is None else torch.cat((self.k, k), dim=2)
        self.v = v if self.v is None else torch.cat((self.v, v), dim=2)
        return self.k, self.v


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.use_rope = cfg.pos == "rope"
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        rope: tuple[Tensor, Tensor] | None = None,
        cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            if rope is None:
                raise ValueError("RoPE enabled but no cos/sin tables were passed")
            cos, sin = rope
            q = apply_rope(q, cos[start_pos : start_pos + T], sin[start_pos : start_pos + T])
            k = apply_rope(k, cos[start_pos : start_pos + T], sin[start_pos : start_pos + T])

        dropout_p = self.dropout if self.training else 0.0

        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        else:
            k_all, v_all = cache.append(k, v)
            k_len = k_all.size(2)
            q_pos = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(1)
            k_pos = torch.arange(k_len, device=x.device).unsqueeze(0)
            mask = k_pos <= q_pos  # (T, k_len) bool; True = attend
            y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask, dropout_p=dropout_p)

        y = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(y))
