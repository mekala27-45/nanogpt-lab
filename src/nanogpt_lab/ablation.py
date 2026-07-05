"""Architecture ablation: train each variant and compare final validation loss.

Isolates one modern component at a time against the Llama-style baseline:
  * learned absolute positions instead of RoPE
  * LayerNorm instead of RMSNorm
  * GELU MLP instead of SwiGLU

Run: ``python -m nanogpt_lab.ablation --config config/smoke.yaml [--max-iters N]``.
Every number is measured on the same data/seed; param counts are reported so
comparisons stay honest (SwiGLU vs GELU differ in parameters).
"""

from __future__ import annotations

import argparse
import json

from nanogpt_lab.config import PROJECT_ROOT, Config, load_config
from nanogpt_lab.logging_setup import configure_logging, get_logger
from nanogpt_lab.train import train

log = get_logger(__name__)

VARIANTS: list[tuple[str, dict[str, str]]] = [
    ("baseline: RoPE + RMSNorm + SwiGLU", {}),
    ("learned positions (no RoPE)", {"pos": "learned"}),
    ("LayerNorm (no RMSNorm)", {"norm": "layer"}),
    ("GELU MLP (no SwiGLU)", {"mlp": "gelu"}),
]


def run_ablation(config: Config, max_iters: int | None = None) -> list[dict[str, float | str]]:
    if max_iters is not None:
        config = config.model_copy(
            update={"train": config.train.model_copy(update={"max_iters": max_iters})}
        )

    results: list[dict[str, float | str]] = []
    for name, override in VARIANTS:
        variant_model = config.model.model_copy(update=override)
        variant = config.model_copy(update={"model": variant_model})
        log.info("ablation.run", variant=name)
        res = train(variant)
        results.append(
            {
                "variant": name,
                "val_loss": round(res.best_val_loss, 4),
                "val_bpc": round(res.val_bpc, 4),
                "n_params": res.n_params,
            }
        )

    baseline = results[0]["val_loss"]
    for r in results:
        assert isinstance(r["val_loss"], float) and isinstance(baseline, float)
        r["delta_vs_baseline"] = round(r["val_loss"] - baseline, 4)

    out = PROJECT_ROOT / "reports" / "ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    log.info("ablation.done", report=str(out))
    print("\n| variant | val loss | val bpc | params | Δ vs baseline |")
    print("|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['variant']} | {r['val_loss']} | {r['val_bpc']} "
            f"| {r['n_params']:,} | {r['delta_vs_baseline']:+} |"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the architecture ablation study.")
    parser.add_argument("--config", default="config/smoke.yaml")
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()

    configure_logging("INFO", "console")
    config = load_config(args.config)
    run_ablation(config, args.max_iters)


if __name__ == "__main__":
    main()
