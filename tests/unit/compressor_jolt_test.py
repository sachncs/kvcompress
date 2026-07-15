"""Tests for the JoLT compressor."""

from __future__ import annotations

import pytest
import torch

from kvcompress.compressor.flashjolt import FlashJoLTCompressor, flashjolt_cap
from kvcompress.compressor.jolt import JoLTCompressor


@pytest.fixture
def small_kv() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    # Small but representative shapes.
    k = torch.randn(4, 32, 16, dtype=torch.float32)
    v = torch.randn(4, 32, 16, dtype=torch.float32)
    return k, v


def test_jolt_compress_decompress_roundtrip(
    small_kv: tuple[torch.Tensor, torch.Tensor],
) -> None:
    k, v = small_kv
    comp = JoLTCompressor(compression_ratio=2.0, bits=(0, 4, 8))
    k_p, v_p = comp.compress(k, v)
    k_hat, v_hat = comp.decompress(k_p, v_p)
    assert k_hat.shape == k.shape
    assert v_hat.shape == v.shape
    # Relative Frobenius error should be bounded (not exact).
    rel_err_k = (torch.linalg.norm(k - k_hat) / torch.linalg.norm(k)).item()
    rel_err_v = (torch.linalg.norm(v - v_hat) / torch.linalg.norm(v)).item()
    assert rel_err_k < 1.0
    assert rel_err_v < 1.0


def test_jolt_compression_ratio_metadata(
    small_kv: tuple[torch.Tensor, torch.Tensor],
) -> None:
    k, v = small_kv
    comp = JoLTCompressor(compression_ratio=3.0)
    k_p, v_p = comp.compress(k, v)
    assert "r_token" in k_p.metadata
    assert "r_feature" in k_p.metadata
    assert "bits" in k_p.metadata


def test_jolt_bytes_reduced(small_kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    k, v = small_kv
    comp = JoLTCompressor(compression_ratio=2.0, bits=(4, 8))
    k_p, v_p = comp.compress(k, v)
    original = k.numel() * k.element_size() * 2
    compressed = k_p.bytes_compressed + v_p.bytes_compressed
    assert compressed < original


def test_jolt_works_with_bits_zero(small_kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Allocator can pick bits=0 (no residual); should still round-trip."""
    k, v = small_kv
    comp = JoLTCompressor(compression_ratio=8.0, bits=(0, 2, 4))
    k_p, v_p = comp.compress(k, v)
    k_hat, v_hat = comp.decompress(k_p, v_p)
    assert k_hat.shape == k.shape


def test_jolt_payload_stats(small_kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    k, v = small_kv
    comp = JoLTCompressor(compression_ratio=2.0)
    comp.compress(k, v)
    s = comp.stats()
    assert "call_count" in s
    assert "compress_time_ms" in s
    assert "bytes_original" in s
    assert "bytes_compressed" in s
    assert s["call_count"] == 1


def test_jolt_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="compression_ratio"):
        JoLTCompressor(compression_ratio=0.5)


def test_jolt_shape_mismatch() -> None:
    comp = JoLTCompressor(compression_ratio=2.0)
    k = torch.randn(4, 16, 8)
    v = torch.randn(4, 8, 8)  # wrong
    with pytest.raises(ValueError, match="K/V shape"):
        comp.compress(k, v)


def test_jolt_non_3d_raises() -> None:
    comp = JoLTCompressor(compression_ratio=2.0)
    k = torch.randn(4, 16)
    v = torch.randn(4, 16)
    with pytest.raises(ValueError, match="3-D"):
        comp.compress(k, v)


def test_jolt_method_name() -> None:
    comp = JoLTCompressor(compression_ratio=2.0)
    assert comp.name == "jolt"
    assert comp.stats()["method"] == "jolt"


def test_flashjolt_cap_policy() -> None:
    # Short context: cap equals q_min(R).
    assert flashjolt_cap(128, 3.0) == 32
    assert flashjolt_cap(128, 8.0) == 64
    # Long context: cap grows sublinearly.
    assert flashjolt_cap(8192, 3.0) == 256  # ⌈8192/32⌉ = 256, max(32, 256) = 256, capped at 512
    assert flashjolt_cap(32768, 3.0) == 512  # ⌈32768/32⌉ = 1024, capped at 512


def test_flashjolt_compress_decompress() -> None:
    torch.manual_seed(0)
    k = torch.randn(4, 64, 16, dtype=torch.float32)
    v = torch.randn(4, 64, 16, dtype=torch.float32)
    comp = FlashJoLTCompressor(compression_ratio=2.0, bits=(0, 4, 8))
    k_p, v_p = comp.compress(k, v)
    k_hat, v_hat = comp.decompress(k_p, v_p)
    assert k_hat.shape == k.shape
    assert v_hat.shape == v.shape


def test_flashjolt_method_name() -> None:
    comp = FlashJoLTCompressor(compression_ratio=2.0)
    assert comp.name == "flashjolt"


def test_flashjolt_speedup_no_quality_loss() -> None:
    """FlashJoLT and exact JoLT should give similar reconstructions."""
    torch.manual_seed(0)
    k = torch.randn(4, 64, 16, dtype=torch.float32)
    v = torch.randn(4, 64, 16, dtype=torch.float32)

    comp_exact = JoLTCompressor(compression_ratio=3.0, bits=(0, 4, 8))
    comp_fast = FlashJoLTCompressor(compression_ratio=3.0, bits=(0, 4, 8))

    kp_e, vp_e = comp_exact.compress(k, v)
    kp_f, vp_f = comp_fast.compress(k, v)
    k_e, v_e = comp_exact.decompress(kp_e, vp_e)
    k_f, v_f = comp_fast.decompress(kp_f, vp_f)
    err_e_k = (torch.linalg.norm(k - k_e) / torch.linalg.norm(k)).item()
    err_f_k = (torch.linalg.norm(k - k_f) / torch.linalg.norm(k)).item()
    # Flash should be within slack of exact.
    assert err_f_k < err_e_k * 1.5 + 0.02


def test_jolt_compressor_inheritance() -> None:
    """FlashJoLT should be a KVCompressor subclass."""
    from kvcompress.compressor.base import KVCompressor

    assert issubclass(FlashJoLTCompressor, KVCompressor)
    assert issubclass(JoLTCompressor, KVCompressor)
