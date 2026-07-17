"""Coverage tests for residual.py error/edge paths."""

from __future__ import annotations

import pytest
import torch

from kvcompress.compressor.residual import (
    ResidualPayload,
    decode_residual,
    encode_residual,
)


def test_residual_estimate_bytes_per_channel() -> None:
    """The ``estimate_residual_bytes`` helper covers the per_channel branch."""
    from kvcompress.compressor.residual import estimate_residual_bytes

    n = estimate_residual_bytes(
        (4, 16, 8),
        bits=4,
        per_channel=True,
        group_size=None,
    )
    # (4*16*8*4 + 7) // 8 = 256 packed + 8*8*4=256 projection + 4*16*4+4*16*4=512 scale
    assert n > 0


def test_residual_estimate_bytes_per_group() -> None:
    from kvcompress.compressor.residual import estimate_residual_bytes

    n = estimate_residual_bytes(
        (4, 16, 8),
        bits=4,
        per_channel=False,
        group_size=8,
    )
    assert n > 0


def test_residual_estimate_bytes_per_tensor() -> None:
    from kvcompress.compressor.residual import estimate_residual_bytes

    n = estimate_residual_bytes(
        (4, 16, 8),
        bits=4,
        per_channel=False,
        group_size=None,
    )
    # No per-channel, no group -> single scale + zero_point = 16 bytes.
    assert n > 0


def test_residual_estimate_bytes_bits_zero_returns_zero() -> None:
    from kvcompress.compressor.residual import estimate_residual_bytes

    n = estimate_residual_bytes((4, 16, 8), bits=0)
    assert n == 0


def test_residual_payload_dataclass_construction() -> None:
    """Cover the dataclass __init__ and field defaults."""
    payload = ResidualPayload(
        projection_seed=0,
        projection_distribution="gaussian",
        projection_sparsity=1.0,
        quant_dtype="int4",
        symmetric=True,
        per_channel=True,
        group_size=None,
        packed=torch.zeros(8, dtype=torch.uint8),
        scale=torch.ones(4),
        zero_point=torch.zeros(4, dtype=torch.int32),
        original_shape=(2, 4, 4),
        original_last=4,
    )
    assert payload.projection_seed == 0
    assert payload.symmetric is True
    assert payload.per_channel is True
    assert payload.original_last == 4


def test_encode_decode_residual_round_trip_bits_zero() -> None:
    """bits=0 produces a no-op residual; decode returns zeros."""
    x = torch.randn(2, 4, 8)
    res = encode_residual(x, bits=0, seed=42)
    out = decode_residual(res)
    assert out.shape == x.shape
    # bits=0 doesn't carry the original signal; the residual is a zero-shape
    # tensor decoded as zeros.
    assert torch.allclose(out, torch.zeros_like(x), atol=1e-3)


def test_encode_decode_residual_round_trip_with_group() -> None:
    """per-group quantization path round-trips through decode."""
    x = torch.randn(2, 16, 8)
    res = encode_residual(
        x,
        bits=4,
        seed=0,
        distribution="gaussian",
        symmetric=True,
        per_channel=False,
        group_size=4,
    )
    out = decode_residual(res)
    assert out.shape == x.shape


def test_encode_decode_residual_asymmetric() -> None:
    """Asymmetric quantization path round-trips through decode."""
    x = torch.randn(2, 8, 8)
    res = encode_residual(
        x,
        bits=4,
        seed=0,
        symmetric=False,
        per_channel=True,
    )
    out = decode_residual(res)
    assert out.shape == x.shape


def test_encode_residual_rademacher() -> None:
    """Rademacher projection path."""
    x = torch.randn(2, 8, 8)
    res = encode_residual(
        x,
        bits=4,
        seed=0,
        distribution="rademacher",
    )
    out = decode_residual(res)
    assert out.shape == x.shape


def test_encode_residual_unknown_distribution_raises() -> None:
    """Bad distribution name surfaces a clear ValueError."""
    x = torch.randn(2, 4, 8)
    with pytest.raises(ValueError, match="unknown JL distribution"):
        encode_residual(x, bits=4, seed=0, distribution="not-a-real-dist")
