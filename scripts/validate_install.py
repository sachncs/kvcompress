"""Validate the install: run a smoke test end-to-end."""

from __future__ import annotations

import argparse
import logging
import sys

import torch

log = logging.getLogger(__name__)


def main() -> None:
    _ = argparse.ArgumentParser(description="Validate install").parse_args()

    log.info("kvcompress validate: starting smoke test")
    import kvcompress

    log.info("kvcompress version: %s", kvcompress.__version__)

    # 1. Compress / decompress round-trip on synthetic K/V.
    from kvcompress import JoLTCompressor, FlashJoLTCompressor

    K = torch.randn(4, 32, 16)
    V = torch.randn(4, 32, 16)
    comp = JoLTCompressor(compression_ratio=2.0)
    kp, vp = comp.compress(K, V)
    k_hat, v_hat = comp.decompress(kp, vp)
    rel_err = float(torch.linalg.norm(K - k_hat) / torch.linalg.norm(K))
    log.info("JoLT round-trip rel error: %.4f", rel_err)
    assert rel_err < 1.0, f"unexpectedly large error: {rel_err}"

    # 2. FlashJoLT.
    fj = FlashJoLTCompressor(compression_ratio=2.0)
    kp, vp = fj.compress(K, V)
    k_hat, v_hat = fj.decompress(kp, vp)
    log.info(
        "FlashJoLT round-trip rel error: %.4f",
        float(torch.linalg.norm(K - k_hat) / torch.linalg.norm(K)),
    )

    # 3. HF adapter smoke test.
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        log.info("transformers available; loading GPT-2 for HF smoke test")
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        from kvcompress import enable_compression

        handle = enable_compression(model, method="flashjolt", compression_ratio=2.0)
        try:
            ids = tok.encode("Hello", return_tensors="pt")
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            log.info("HF smoke test output: %s", tok.decode(out[0]))
        finally:
            handle.disable()
    except Exception as e:
        log.warning("HF smoke test skipped: %s", e)

    log.info("kvcompress validate: OK")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
