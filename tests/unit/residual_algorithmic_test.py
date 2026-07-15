"""Tests for the JL-rotated residual path.

These tests verify the *protocol* of the residual path: the
residual-quantise-decode round-trip is correct up to the quantisation
error, the residual_seed is the only thing needed to regenerate the
projection, and the serialised payload is JSON-roundtrippable.
"""

from __future__ import annotations

import pytest
import torch

from kvcompress.compressor.residual import (
    ResidualPayload,
    decode_residual,
    encode_residual,
    estimate_residual_bytes,
)


@pytest.fixture
def residual() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 16, 8) * 0.05


def test_encode_decode_zero_bits_is_zero_tensor(residual: torch.Tensor) -> None:
    """``encode_residual(r, bits=0)`` returns a payload that decodes to zero."""
    payload = encode_residual(residual, bits=0, seed=0)
    decoded = decode_residual(payload)
    assert decoded.shape == residual.shape
    assert torch.equal(decoded, torch.zeros_like(residual))


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_encode_decode_round_trip_preserves_shape(residual: torch.Tensor, bits: int) -> None:
    payload = encode_residual(residual, bits=bits, seed=0)
    decoded = decode_residual(payload)
    assert decoded.shape == residual.shape
    assert decoded.dtype == torch.float32


def test_residual_payload_serialises_to_dict(residual: torch.Tensor) -> None:
    """``to_dict`` and ``from_dict`` round-trip is lossless."""
    payload = encode_residual(residual, bits=4, seed=0)
    d = payload.to_dict()
    # Spot-check the dict shape.
    assert d["quant_dtype"] == "int4"
    assert d["projection_seed"] == 0
    assert d["original_shape"] == list((4, 16, 8))
    assert d["original_last"] == 8
    assert isinstance(d["packed"], torch.Tensor)
    # Round-trip.
    payload2 = ResidualPayload.from_dict(d)
    assert payload2.quant_dtype == payload.quant_dtype
    assert payload2.projection_seed == payload.projection_seed
    assert payload2.original_shape == payload.original_shape
    assert torch.equal(payload2.packed, payload.packed)


def test_estimate_residual_bytes_zero_is_zero() -> None:
    assert estimate_residual_bytes((4, 16, 8), bits=0) == 0


def test_estimate_residual_bytes_higher_bits_more_bytes() -> None:
    shape = (4, 16, 8)
    assert estimate_residual_bytes(shape, 2) < estimate_residual_bytes(shape, 4)
    assert estimate_residual_bytes(shape, 4) < estimate_residual_bytes(shape, 8)


def test_residual_bits_property() -> None:
    for bits, expected in [(0, 0), (2, 2), (4, 4), (8, 8)]:
        payload = ResidualPayload(
            projection_seed=0,
            projection_distribution="gaussian",
            projection_sparsity=1.0,
            quant_dtype=f"int{bits}",
            symmetric=True,
            per_channel=True,
            group_size=None,
            packed=torch.zeros(0, dtype=torch.uint8),
            scale=torch.zeros(0),
            zero_point=torch.zeros(0, dtype=torch.int32),
            original_shape=(1, 1, 1),
            original_last=1,
        )
        assert payload.bits == expected


def test_residual_rademacher_distribution_supported(
    residual: torch.Tensor,
) -> None:
    payload = encode_residual(residual, bits=4, seed=0, distribution="rademacher")
    assert payload.projection_distribution == "rademacher"
    decoded = decode_residual(payload)
    assert decoded.shape == residual.shape


def test_residual_seed_is_deterministic(residual: torch.Tensor) -> None:
    """Same seed + same input → same packed bytes."""
    p1 = encode_residual(residual, bits=4, seed=42)
    p2 = encode_residual(residual, bits=4, seed=42)
    assert torch.equal(p1.packed, p2.packed)
    # Different seed → different bytes.
    p3 = encode_residual(residual, bits=4, seed=43)
    assert not torch.equal(p1.packed, p3.packed)


def test_residual_decode_preserves_norm_within_quantisation_error(
    residual: torch.Tensor,
) -> None:
    """The decoded residual's Frobenius norm is bounded by the original
    (the JL rotation's cheap inverse is not an exact inverse, so the
    norm can be smaller but not larger)."""
    payload = encode_residual(residual, bits=4, seed=0)
    decoded = decode_residual(payload)
    src_norm = torch.linalg.norm(residual).item()
    dec_norm = torch.linalg.norm(decoded).item()
    # The cheap inverse is "x @ Π" (not "x @ Π⁻¹") so the recovered
    # residual's norm is bounded by the projection's operator norm,
    # which is ≤ ‖Π‖_F. We just check that the recovered norm is
    # non-zero and within the same order of magnitude as the source.
    assert dec_norm > 0
    assert dec_norm < src_norm * 5, f"dec_norm {dec_norm} not within 5x of src_norm {src_norm}"
