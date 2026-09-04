"""Long-context needle-in-haystack benchmark.

Tests how well a model with compression enabled can still retrieve a
specific fact ("needle") buried in a long context. Uses a small model
(GPT-2 by default) and a synthetic needle.

This is a smoke test for long-context behaviour; the paper's full
RULER evaluation is left for future work.
"""

from __future__ import annotations

import argparse
import logging
import random

import torch

log = logging.getLogger(__name__)


def make_needle_context(needle: str, context_length: int, seed: int = 0) -> tuple[str, int]:
    """Return (full_text, position_of_needle) for a context of ~``context_length`` tokens."""
    rng = random.Random(seed)
    filler = "The grass is green. The sky is blue. The sun is yellow. "
    # Build until we exceed the target token count.
    chunks = []
    while sum(len(c.split()) for c in chunks) < context_length:
        chunks.append(filler)
    full = " ".join(chunks)
    words = full.split()
    # Insert the needle at a random position.
    pos = rng.randint(0, max(0, len(words) - 1))
    needle_words = needle.split()
    words = words[:pos] + needle_words + words[pos:]
    return " ".join(words), pos


def main() -> None:
    parser = argparse.ArgumentParser(description="Needle-in-haystack benchmark")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--needle", type=str, default="The secret code is 12345.")
    parser.add_argument("--max-new", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("loading model: %s", args.model)
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.eval()
    except Exception as e:
        log.warning("could not load model: %s", e)
        return

    context, pos = make_needle_context(args.needle, args.context_length, seed=args.seed)
    log.info("context length ~%d tokens; needle at position %d", len(context.split()), pos)

    prompt = f"{context}\n\nQuestion: What is the secret code?\nAnswer:"
    ids = tok.encode(prompt, return_tensors="pt")
    log.info("prompt tokens: %d", ids.shape[1])

    from kvfold import enable_compression

    for method in ("identity", "flashjolt"):
        handle = enable_compression(model, method=method, compression_ratio=3.0)
        try:
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=args.max_new,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
            log.info("[%s] generated: %s", method, text)
            log.info("[%s] stats: %s", method, handle.stats_dict())
        finally:
            handle.disable()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
