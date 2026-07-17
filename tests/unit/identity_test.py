"""Tests for :class:`IdentityCompressor`.

Dedicated tests for the "passthrough" compressor that the dispatch wires
to ``method="identity"``, ``"fp16"``, ``"bf16"``, and ``"fp8"``.
"""

from __future__ import annotations

import pytest
import torch

from kvcompress import IdentityCompressor


@pytest.fixture
def k_v() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(2, 8, 16), torch.randn(2, 8, 16)


def test_identity_preserves_shape(k_v: tuple[torch.Tensor, torch.Tensor]) -> None:
    k, v = k_v
    c = IdentityCompressor()
    kp, vp = c.compress(k, v)
    assert kp.shape == k.shape
    assert vp.shape == v.shape


def test_identity_dtype_cast_to_factor(k_v: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Default factor_dtype is fp16; stored K/V values are fp16 even when input is fp32."""
    k, v = k_v
    c = IdentityCompressor()
    kp, _ = c.compress(k, v)
    assert kp.data["value"].dtype == torch.float16


def test_identity_factor_dtype_override(k_v: tuple[torch.Tensor, torch.Tensor]) -> None:
    """factor_dtype=bf16 stores values in bf16."""
    k, v = k_v
    c = IdentityCompressor(factor_dtype=torch.bfloat16)
    kp, _ = c.compress(k, v)
    assert kp.data["value"].dtype == torch.bfloat16


def test_identity_factor_dtype_fp32_roundtrip(k_v: tuple[torch.Tensor, torch.Tensor]) -> None:
    """factor_dtype=fp32 makes the round-trip bit-exact."""
    k, v = k_v
    c = IdentityCompressor(factor_dtype=torch.float32)
    kp, vp = c.compress(k, v)
    k_hat, v_hat = c.decompress(kp, vp)
    assert torch.equal(k_hat, k)
    assert torch.equal(v_hat, v)


def test_identity_dtype_restore_on_decompress(k_v: tuple[torch.Tensor, torch.Tensor]) -> None:
    """decompress returns the original input dtype, not the storage dtype."""
    k = k_v[0].to(torch.float32)
    c = IdentityCompressor(factor_dtype=torch.float16)
    kp, _ = c.compress(k, k)
    k_hat, _ = c.decompress(kp, kp)
    assert k_hat.dtype == torch.float32


def test_identity_kv_separation(k_v: tuple[torch.Tensor, torch.Tensor]) -> None:
    """K and V payloads must be distinct — the bug this used to carry
    was compressing one side and storing it under both keys.
    """
    k, v = k_v
    c = IdentityCompressor(factor_dtype=torch.float32)
    kp, vp = c.compress(k, v)
    assert not torch.equal(kp.data["value"], vp.data["value"])
    k_hat, v_hat = c.decompress(kp, vp)
    assert torch.equal(k_hat, k)
    assert torch.equal(v_hat, v)
    assert not torch.equal(k_hat, v_hat)


def test_identity_stats_correct(k_v: tuple[torch.Tensor, torch.Tensor]) -> None:
    """``bytes_compressed`` reflects the factor_dtype's element size."""
    k, v = k_v
    c = IdentityCompressor(factor_dtype=torch.float16)
    kp, _ = c.compress(k, v)
    assert kp.stats.bytes_compressed == k.numel() * 2  # fp16 = 2 bytes
    assert kp.stats.bytes_original == k.numel() * k.element_size()


def test_identity_extra_kwargs_are_dropped(k_v: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Unknown kwargs are silently dropped at the dispatch boundary;
    the compressor itself doesn't validate."""
    k, v = k_v
    c = IdentityCompressor(factor_dtype=torch.float32, random_unused_arg=42)
    kp, vp = c.compress(k, v)
    assert kp.shape == k.shape
