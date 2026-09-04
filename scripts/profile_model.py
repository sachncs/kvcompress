"""Profile model: per-layer compress/decompress timings."""

from __future__ import annotations

import argparse
import logging

import torch

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a model")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--ratio", type=float, default=3.0)
    parser.add_argument("--max-new", type=int, default=30)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.eval()
    except Exception as e:
        log.warning("could not load model: %s", e)
        return

    from kvfold import enable_compression

    handle = enable_compression(model, method="flashjolt", compression_ratio=args.ratio)
    try:
        ids = tok.encode("The quick brown fox jumps over the lazy dog", return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        log.info("output: %s", tok.decode(out[0]))
        log.info("stats: %s", handle.stats_dict())
    finally:
        handle.disable()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
