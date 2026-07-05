# Architecture & Design Decisions

NanoGPT-Lab is a decoder-only transformer written from scratch in PyTorch. The
default configuration is deliberately **Llama-style** rather than GPT-2-style,
and every modern component is a config toggle so the ablation study can measure
each one's contribution in isolation.

## The forward pass

```
tokens ─► embedding ─► [ pre-norm block ] × N ─► final norm ─► tied LM head ─► logits
                          │
                          ├─ x = x + Attention(Norm(x))   (RoPE inside attention)
                          └─ x = x + MLP(Norm(x))
```

There is no learned positional embedding table in the default config — position
is injected *inside attention* by RoPE. Norm is RMSNorm; the MLP is SwiGLU; the
LM head shares weights with the token embedding.

## Component decisions

### RoPE (rotary positional embeddings)
Instead of adding a learned position vector, RoPE **rotates** each query and key
by an angle proportional to its absolute position. Because a dot product of two
rotated vectors depends only on the *difference* of their angles, attention
scores become a function of **relative** position — which generalizes better and
needs no parameters. Implemented in `model/rope.py` with the GPT-NeoX
"rotate-half" convention; `tests/test_rope.py` verifies both the norm-preserving
(it's a rotation) and relative-position properties.

### RMSNorm
Root-mean-square norm drops the mean-centering and bias of LayerNorm, keeping
only the rescaling. It's cheaper and, empirically (see the ablation), at least as
good here. ~`dim` fewer parameters per norm and one fewer reduction.

### SwiGLU MLP
`(SiLU(xW1) ⊙ xW3) W2` — a gated activation. It uses three projections instead
of two, so for a fair comparison the ablation reports parameter counts alongside
loss.

### Weight tying
The output projection reuses the token-embedding matrix (`lm_head.weight is
tok_emb.weight`). Fewer parameters, a well-known regularizer for LMs, and it
means the input and output live in the same space.

### KV cache — and why the test matters most
During generation, recomputing attention over the whole prefix every step is
`O(T²)` wasted work. The KV cache stores past keys/values so each new token is
`O(T)`. But a cache is a classic source of subtle bugs: get the RoPE position or
the attention mask off by one and generations silently degrade.

`tests/test_kv_cache.py` pins this down: it asserts that **one-token-at-a-time
cached decoding produces logits identical (atol 1e-4) to a single full-sequence
forward pass**, across all five architecture variants. If the cache math were
wrong, that test would fail. This is the single most important correctness check
in the repo.

## Training

- **Optimizer:** AdamW (β = 0.9, 0.95). Weight decay applies to 2-D tensors
  (matmuls/embeddings) but not to norms or biases — the standard split.
- **Schedule:** linear warmup then cosine decay to `min_lr`.
- **Stability:** global grad-norm clipping.
- **Checkpointing:** best-validation-loss checkpoint is saved with the config and
  tokenizer embedded, so `generate` / the Gradio app can reload it standalone.
- **Metrics:** validation cross-entropy, reported also as **bits-per-character**
  (loss / ln 2) and perplexity.

## Attention implementation notes

`model/attention.py` has one implementation with two paths:
- **Training / full forward** — no cache, `is_causal=True` so PyTorch uses the
  fused (flash) scaled-dot-product-attention kernel.
- **Incremental decode** — with a cache, an explicit boolean mask lets the new
  queries attend to all cached keys up to their own absolute position (works for
  both a single-token step and a multi-token prefill).

## Compute strategy

The `smoke` config trains a ~1M-parameter model on Tiny Shakespeare and runs on
**CPU in a few minutes** — it's the repo's reproducible default and what CI
exercises (via the correctness tests). The `full` config (~25M params, TinyStories,
`torch.compile`) targets a GPU; run it on Colab/Kaggle. Same code path, one flag.

## Deliberate non-goals

- No custom CUDA kernels — the point is a *correct, readable* implementation.
- No BPE tokenizer — char-level keeps the vocab tiny and the focus on the model.
  BPE is the obvious next step (see [INTERVIEW.md](INTERVIEW.md)).
- No distributed training — single-device is enough at this scale.
