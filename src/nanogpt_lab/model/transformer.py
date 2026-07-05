"""The GPT model: token embedding → N pre-norm blocks → tied LM head.

Architecture is Llama-style by default (RoPE, RMSNorm, SwiGLU, weight tying),
but every component is swappable via config so the ablation study can isolate
each one's contribution.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanogpt_lab.config import ModelCfg
from nanogpt_lab.model.attention import CausalSelfAttention, KVCache
from nanogpt_lab.model.layers import make_mlp, make_norm
from nanogpt_lab.model.rope import build_rope_cache


class Block(nn.Module):
    """Pre-norm residual block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.norm1 = make_norm(cfg.norm, cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = make_norm(cfg.norm, cfg.n_embd)
        self.mlp = make_mlp(cfg.mlp, cfg.n_embd, cfg.mlp_ratio, cfg.dropout)

    def forward(
        self,
        x: Tensor,
        rope: tuple[Tensor, Tensor] | None,
        cache: KVCache | None,
        start_pos: int,
    ) -> Tensor:
        x = x + self.attn(self.norm1(x), rope, cache, start_pos)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelCfg, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.block_size = cfg.block_size

        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = make_norm(cfg.norm, cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        if cfg.pos == "rope":
            cos, sin = build_rope_cache(cfg.block_size, cfg.head_dim, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual output projections.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_emb is not None:
            n -= self.pos_emb.weight.numel()
        return int(n)

    def _rope(self) -> tuple[Tensor, Tensor] | None:
        if self.cfg.pos == "rope":
            return (self.get_buffer("rope_cos"), self.get_buffer("rope_sin"))
        return None

    def new_caches(self) -> list[KVCache]:
        return [KVCache() for _ in self.blocks]

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        caches: list[KVCache] | None = None,
        start_pos: int = 0,
    ) -> tuple[Tensor, Tensor | None]:
        _, T = idx.shape
        if start_pos + T > self.block_size:
            raise ValueError(
                f"position {start_pos}+{T} exceeds block_size {self.block_size}; crop the context"
            )
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(start_pos, start_pos + T, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        rope = self._rope()
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x = block(x, rope, cache, start_pos)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss: Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss
