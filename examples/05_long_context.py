"""05 — Long-context smoke test.

Loads a small model, feeds it a long prompt with a needle buried at a
random position, and checks that compression doesn't break the
generation pipeline.
"""

from __future__ import annotations

import logging
import random

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from kvcompress import enable_compression

log = logging.getLogger(__name__)


def make_needle_context(needle: str, context_length: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    filler = "The grass is green. The sky is blue. The sun is yellow. "
    chunks: list[str] = []
    while sum(len(c.split()) for c in chunks) < context_length:
        chunks.append(filler)
    full = " ".join(chunks)
    words = full.split()
    pos = rng.randint(0, max(0, len(words) - 1))
    return " ".join(words[:pos] + needle.split() + words[pos:])


def main() -> None:
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    needle = "The secret code is 12345."
    context = make_needle_context(needle, context_length=512)
    prompt = f"{context}\n\nQuestion: What is the secret code?\nAnswer:"
    ids = tok.encode(prompt, return_tensors="pt")
    log.info("prompt tokens: %d", ids.shape[1])

    for method in ("identity", "flashjolt"):
        handle = enable_compression(model, method=method, target_memory="33%")
        try:
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
            print(f"[{method}] {text}")
        finally:
            handle.disable()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
