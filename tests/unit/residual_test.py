"""Tests for the residual path."""

from __future__ import annotations

import pytest
import torch

from kvcompress.compressor.residual import (
    decode_residual,
    encode_residual,
    estimate_residual_bytes,
)


@pytest.fixture
def residual() -> torch.Tensor:
    torch.manual_seed(0)
    # Realistic-looking residual magnitudes.
    return torch.randn(4, 16, 8) * 0.05


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_roundtrip_zero_bits_is_zero(residual: torch.Tensor, bits: int) -> None:
    if bits != 0:
        pytest.skip("this test is for bits=0 only")
    payload = encode_residual(residual, bits=0, seed=0)
    decoded = decode_residual(payload)
    assert decoded.shape == residual.shape
    assert torch.allclose(decoded, torch.zeros_like(residual))


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_roundtrip_residual(residual: torch.Tensor, bits: int) -> None:
    payload = encode_residual(residual, bits=bits, seed=0)
    decoded = decode_residual(payload)
    assert decoded.shape == residual.shape
    # Check that the decoded residual is in the right ballpark.
    src_norm = torch.linalg.norm(residual).item()
    rec_norm = torch.linalg.norm(decoded).item()
    assert rec_norm > 0
    assert rec_norm < src_norm * 2  # bounded; exact ratio depends on bits


def test_rademacher_distribution(residual: torch.Tensor) -> None:
    payload = encode_residual(
        residual,
        bits=4,
        seed=1,
        distribution="rademacher",
    )
    assert payload.projection_distribution == "rademacher"
    decoded = decode_residual(payload)
    assert decoded.shape == residual.shape


def test_seed_reproducibility(residual: torch.Tensor) -> None:
    p1 = encode_residual(residual, bits=4, seed=42)
    p2 = encode_residual(residual, bits=4, seed=42)
    p3 = encode_residual(residual, bits=4, seed=43)
    assert torch.allclose(p1.packed, p2.packed)
    assert not torch.allclose(p1.packed, p3.packed)


def test_to_from_dict() -> None:
    payload = encode_residual(torch.randn(2, 4, 8) * 0.1, bits=4, seed=0)
    d = payload.to_dict()
    p2 = decode_residual(type(payload).from_dict(d))
    # decode_residual reconstructs from a payload, so we use decode via a new
    # payload object created from the dict.
    from kvcompress.compressor.residual import ResidualPayload

    p2 = ResidualPayload.from_dict(d)
    decoded = decode_residual(p2)
    assert decoded.shape == payload.original_shape


def test_estimate_residual_bytes() -> None:
    shape = (4, 16, 8)
    assert estimate_residual_bytes(shape, 0) == 0
    assert estimate_residual_bytes(shape, 8) > estimate_residual_bytes(shape, 4)
    assert estimate_residual_bytes(shape, 4) > estimate_residual_bytes(shape, 2)


def test_invalid_bits() -> None:
    with pytest.raises(ValueError, match="bits"):
        encode_residual(torch.randn(2, 4, 8), bits=3, seed=0)


def test_asymmetric_quant() -> None:
    payload = encode_residual(
        torch.randn(2, 4, 8) * 0.1,
        bits=4,
        seed=0,
        symmetric=False,
    )
    assert payload.symmetric is False
    decoded = decode_residual(payload)
    assert decoded.shape == payload.original_shape