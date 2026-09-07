"""01 — Quickstart.

The smallest possible example of using kvcompress with a Hugging Face
model. Loads GPT-2, enables 3x compression, generates a short
continuation, and prints stats.
"""

from __future__ import annotations

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from kvcompress import enable_compression


def main() -> None:
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    handle = enable_compression(model, method="flashjolt", compression_ratio=3.0)

    try:
        ids = tok.encode("The capital of France is", return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=15, do_sample=False, pad_token_id=tok.eos_token_id
            )
        print(tok.decode(out[0]))
        print("stats:", handle.stats_dict())
    finally:
        handle.disable()


if __name__ == "__main__":
    main()
