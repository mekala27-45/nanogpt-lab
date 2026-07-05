# Interview Prep — NanoGPT-Lab

Your cheat sheet for defending a from-scratch transformer in a technical screen.

## The 30-second pitch

> "NanoGPT-Lab is a decoder-only transformer I wrote from scratch — no
> `nn.Transformer`, no HuggingFace model. It's Llama-style: RoPE for position,
> RMSNorm, a SwiGLU MLP, weight tying, and a KV cache for fast generation. Every
> component is a config flag, so I ran an ablation that measures what each one
> actually buys you. The part I'm proudest of is a test that proves the KV-cache
> decoding is bit-for-bit equivalent to a full forward pass — that's the bug that
> usually ships silently."

## Know the architecture cold

| Piece | What it does | Why it's better than the GPT-2 default |
|---|---|---|
| **RoPE** | rotates q/k by an angle ∝ position | attention depends on *relative* position; zero parameters; extrapolates |
| **RMSNorm** | rescales by RMS, no mean/bias | cheaper; matches LayerNorm quality here |
| **SwiGLU** | gated `(SiLU(xW1)⊙xW3)W2` | stronger MLP capacity per FLOP |
| **Weight tying** | share embedding ↔ LM head | fewer params; regularizes; shared space |
| **KV cache** | store past k/v during decode | O(T) per step instead of O(T²) |

## The hardest problem I solved

**Proving the KV cache is correct.** During generation you feed one token at a
time and reuse cached keys/values, applying RoPE at the *current absolute
position* and masking so the new query sees all past keys. It is extremely easy
to be off by one on the position or the mask and get generations that look
plausible but are subtly wrong — the kind of bug that never throws.

I made it falsifiable: `test_kv_cache.py` runs a full-sequence forward and a
one-token-at-a-time cached decode on the same input and asserts the logits match
to `1e-4`, across all five architecture variants. Green means the RoPE offsets
and the cache mask are provably right.

## Tradeoffs I'd defend

- **Llama-style over GPT-2-style.** RoPE + RMSNorm + SwiGLU is the modern default
  for a reason; the ablation shows it, rather than me asserting it.
- **Char-level tokenizer.** Keeps the vocab at 65 and the focus on the model. The
  cost is longer sequences per unit of text; BPE is the upgrade.
- **`scaled_dot_product_attention` instead of hand-rolled softmax attention.** I
  *understand* the QKᵀ/√d · softmax · V math (and the causal mask), but calling
  the fused kernel is the right engineering call — it's flash-attention when
  available. The from-scratch depth is in RoPE, the norms, the MLP, and the cache.
- **CPU smoke + GPU full.** Same code, one config flag. The repo stays runnable by
  anyone in minutes; the big run is a Colab notebook.

## Questions I expect

**"Walk me through attention."**
Project x to Q, K, V. Scores = QKᵀ/√d, mask out the future (causal), softmax over
keys, weight V. Multi-head = do it in parallel subspaces and concat. RoPE rotates
Q and K before the scores so the dot product encodes relative position.

**"Why does RoPE encode *relative* position?"**
A rotation by angle θ_m on q and θ_n on k makes their dot product depend on
θ_m − θ_n, i.e. (m − n). Absolute rotations cancel. `test_rope_is_relative`
checks exactly this.

**"Why divide by √d in attention?"**
Without it, dot products grow with dimension, pushing softmax into saturated
regions with vanishing gradients. √d keeps the variance ~1.

**"How do you know your model actually learns?"**
Two ways: `test_overfit_single_batch` drives loss on a fixed batch to ~0 (gradients
flow, capacity exists), and the real run drops validation bits-per-char far below
the log₂(vocab) random baseline while producing recognizable text.

**"What's the compute/memory cost of the KV cache?"**
Memory grows linearly with sequence length × layers × heads × head_dim. It trades
memory for time — worth it for autoregressive decoding.

## Measured results

- 1.06M-parameter char model, Tiny Shakespeare, **CPU**, 2,000 iters.
- **Validation: loss 1.506 · 2.17 bits/char · perplexity 4.51.**
- Random baseline is log₂(65) ≈ 6.02 bits/char, so the model cuts per-character
  uncertainty **from 6.02 to 2.17 bits — a ~64% reduction** — and generates text
  with correct Shakespearean structure (character cues, dialogue, archaic diction).
- The ablation (see [README](README.md)) quantifies each component on identical
  data/seed: **RoPE is the biggest win** (+0.31 loss when replaced by learned
  positions), **RMSNorm ties LayerNorm** but is cheaper, and **SwiGLU edges GELU**
  by +0.11 (partly because it has more parameters — the honest caveat).

## What I'd do with more time

1. **BPE tokenizer** — shorter sequences, real word structure.
2. **Grouped-query attention (GQA)** — cheaper KV cache for bigger models.
3. **`torch.compile` + mixed precision** benchmarks — throughput before/after.
4. **Longer-context tricks** — RoPE scaling / sliding-window attention.
5. **A proper eval** — held-out perplexity vs a HuggingFace baseline of equal size.
6. **Speculative decoding** using the small model as a drafter.
