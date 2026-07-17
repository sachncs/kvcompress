"""Sweep compression ratios for a fixed model and report memory + latency."""

from __future__ import annotations

import argparse
import logging
import time

import torch

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep compression ratios")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.5, 2.0, 3.0, 4.0])
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["flashjolt"],
        help="Compression methods to sweep. Defaults to flashjolt.",
    )
    parser.add_argument("--max-new", type=int, default=20)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.eval()
    except Exception as e:
        log.warning("could not load model: %s", e)
        return

    prompt_ids = torch.randint(0, tok.vocab_size, (1, args.prompt_tokens))
    from kvcompress import enable_compression

    print(f"{'method':<12} {'ratio':>8} {'tokens':>10} {'ms/tok':>10}")
    print("-" * 50)
    for ratio in args.ratios:
        for method in args.methods:
            handle = enable_compression(model, method=method, compression_ratio=ratio)
            try:
                with torch.no_grad():
                    t0 = time.perf_counter()
                    out = model.generate(
                        prompt_ids,
                        max_new_tokens=args.max_new,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                    elapsed = (time.perf_counter() - t0) * 1000
                tokens = out.shape[1] - prompt_ids.shape[1]
                ms_per_tok = elapsed / max(1, tokens)
                print(f"{method:<12} {ratio:>8.2f} {tokens:>10} {ms_per_tok:>9.2f}")
            finally:
                handle.disable()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
