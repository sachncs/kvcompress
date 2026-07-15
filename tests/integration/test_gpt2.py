"""Integration test: enable_compression on GPT-2."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.integration
def test_gpt2_identity_matches_baseline(gpt2_model_with_pad) -> None:
    """With identity compressor, output should match uncompressed exactly."""
    from kvcompress import enable_compression

    tok, model = gpt2_model_with_pad
    ids = tok.encode("The quick brown fox", return_tensors="pt")

    with torch.no_grad():
        baseline = model.generate(
            ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id
        )

    handle = enable_compression(model, method="identity", target_memory="100%")
    try:
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id
            )
        assert torch.equal(baseline, out), "identity should produce identical output"
    finally:
        handle.disable()


@pytest.mark.integration
def test_gpt2_flashjolt_runs(gpt2_model_with_pad) -> None:
    """FlashJoLT compression on GPT-2 should run without error."""
    from kvcompress import enable_compression

    tok, model = gpt2_model_with_pad
    ids = tok.encode("The quick brown fox", return_tensors="pt")

    handle = enable_compression(model, method="flashjolt", compression_ratio=3.0)
    try:
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        assert out.shape[1] == ids.shape[1] + 15
        s = handle.stats_dict()
        assert s["compress_calls"] > 0
        assert s["bytes_original"] > 0
        assert s["bytes_compressed"] > 0
    finally:
        handle.disable()


@pytest.mark.integration
def test_gpt2_disable_restores_behavior(gpt2_model_with_pad) -> None:
    """After disable, output should be identical to baseline."""
    from kvcompress import enable_compression

    tok, model = gpt2_model_with_pad
    ids = tok.encode("Hello world", return_tensors="pt")

    handle = enable_compression(model, method="flashjolt", compression_ratio=2.0)
    with torch.no_grad():
        out_compressed = model.generate(
            ids, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id
        )
    handle.disable()

    with torch.no_grad():
        out_baseline = model.generate(
            ids, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id
        )

    assert torch.equal(
        out_compressed, out_baseline
    ), "after disable, output should be identical to baseline"


@pytest.mark.integration
def test_gpt2_method_switch(gpt2_model_with_pad) -> None:
    """Switching between methods should be supported."""
    from kvcompress import enable_compression

    tok, model = gpt2_model_with_pad
    ids = tok.encode("Hello", return_tensors="pt")

    for method in ("identity", "flashjolt"):
        handle = enable_compression(model, method=method, compression_ratio=3.0)
        try:
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=3,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            assert out.shape[1] == ids.shape[1] + 3
        finally:
            handle.disable()
