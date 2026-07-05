"""Gradio demo for NanoGPT-Lab — deployable to Hugging Face Spaces (Gradio SDK).

Loads a trained checkpoint and streams generated text with temperature / top-k
controls. Point NANOGPT_CHECKPOINT at a checkpoint (default: checkpoints/best.pt).

    pip install -r requirements-demo.txt
    python app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch

from nanogpt_lab.model.generate import generate, load_checkpoint

CHECKPOINT = os.environ.get("NANOGPT_CHECKPOINT", "checkpoints/best.pt")
DEVICE = "cpu"

model, tokenizer, config = load_checkpoint(Path(CHECKPOINT), DEVICE)


def run(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    ids = torch.tensor([tokenizer.encode(prompt) or [0]], dtype=torch.long, device=DEVICE)
    out = generate(model, ids, int(max_new_tokens), float(temperature), int(top_k))
    return tokenizer.decode(out[0].tolist())


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(label="Prompt", value=config.generate.prompt, lines=2),
        gr.Slider(16, 800, value=300, step=16, label="Max new tokens"),
        gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature"),
        gr.Slider(1, 100, value=50, step=1, label="Top-k"),
    ],
    outputs=gr.Textbox(label="Generated text", lines=16),
    title="NanoGPT-Lab — a Llama-style transformer built from scratch",
    description=(
        "RoPE · RMSNorm · SwiGLU · weight tying · KV-cache — no `nn.Transformer`. "
        "Trained character-level; generation crops context to the model's block size."
    ),
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
