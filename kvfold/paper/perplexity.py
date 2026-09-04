"""Table 1 perplexity sweep (paper reproduction).

Loads an HF model, enables compression at each target ratio, computes
WikiText-103 (or wikitext-2-raw-v1 if not available) perplexity, and
records the result.

Requires the ``paper`` optional extra (``pip install kvfold[paper]``).

Usage::

    python -m kvfold.paper.perplexity --model gpt2 --ratios 1 2 3 4 5 8 \
        --output results/perplexity.json

The output is a JSON file with one entry per ratio:

.. code-block:: json

    [
      {"ratio": 1.0, "perplexity": 29.0, "wall_ms": 1234.5, "method": "jolt"},
      ...
    ]
"""

from __future__ import annotations

import logging
import time

import torch

log = logging.getLogger(__name__)


def wikitext_load(sequence_length: int = 1024, split: str = "test") -> list[torch.Tensor]:
    """Load WikiText tokens as a list of fixed-length tensors."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]
    n_chunks = input_ids.numel() // sequence_length
    chunks = [
        input_ids[i * sequence_length : (i + 1) * sequence_length].long()
        for i in range(n_chunks)
    ]
    return chunks


def perplexity(model: torch.nn.Module, chunks: list[torch.Tensor], device: torch.device) -> float:
    """Compute perplexity over ``chunks``.

    Uses the standard ``exp(mean(loss))`` formula. Loss is computed with the
    model's built-in HF forward.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.unsqueeze(0).to(device)
            out = model(chunk, labels=chunk)
            loss = float(out.loss)
            total_loss += loss * chunk.numel()
            total_tokens += chunk.numel()
    avg_loss = total_loss / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(avg_loss)).item())


def run_table1_perplexity(
    *,
    model_name: str = "gpt2",
    ratios: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 8.0),
    sequence_length: int = 1024,
    method: str = "jolt",
    seed: int = 0,
) -> list[dict[str, float]]:
    """Run the full Table 1 sweep and return one row per ratio."""
    from transformers import AutoModelForCausalLM

    from kvfold.api import CompressionHandle, build_compressor, disable_compression, enable_compression

    log.info("Loading model %s and WikiText tokens", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = next(model.parameters()).device
    chunks = wikitext_load(sequence_length=sequence_length)
    log.info("Loaded %d chunks of length %d", len(chunks), sequence_length)

    rows: list[dict[str, float]] = []
    base_perp = perplexity(model, chunks, device)
    rows.append(
        {
            "ratio": 1.0,
            "method": "identity",
            "perplexity": float(base_perp),
            "wall_ms": 0.0,
        }
    )

    handle: CompressionHandle | None = None
    for ratio in ratios:
        if ratio == 1.0:
            continue
        log.info("Enabling compression ratio=%s method=%s", ratio, method)
        if handle is not None:
            disable_compression(handle)
        t0 = time.perf_counter()
        handle = enable_compression(model, method=method, compression_ratio=ratio, seed=seed)
        perp = perplexity(model, chunks, device)
        wall_ms = (time.perf_counter() - t0) * 1000
        rows.append(
            {
                "ratio": float(ratio),
                "method": method,
                "perplexity": float(perp),
                "wall_ms": float(wall_ms),
            }
        )
    if handle is not None:
        disable_compression(handle)
    return rows


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Table 1 perplexity sweep")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0, 8.0])
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--method", type=str, default="jolt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rows = run_table1_perplexity(
        model_name=args.model,
        ratios=tuple(args.ratios),
        sequence_length=args.length,
        method=args.method,
        seed=args.seed,
    )
    for row in rows:
        print(f"ratio={row['ratio']:>5}  method={row['method']:<10}  ppl={row['perplexity']:.3f}  wall={row['wall_ms']:.1f}ms")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)


__all__ = ["wikitext_load", "perplexity", "run_table1_perplexity", "main"]
