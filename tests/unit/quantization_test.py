"""Tests for the quantization module."""

from __future__ import annotations

import pytest
import torch

from kvcompress.compressor.quantization import (
    IntQuantizer,
    dequantize_tensor,
    estimate_int_bytes,
    get_quantizer,
    quantize_tensor,
)


@pytest.fixture
def tensor() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(8, 16, dtype=torch.float32) * 4.0


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_int_symmetric_roundtrip(tensor: torch.Tensor, bits: int) -> None:
    q = IntQuantizer(bits=bits, symmetric=True, per_channel=True)
    packed, scale, zp = q.quantize(tensor)
    x_hat = q.dequantize(packed, scale, zp, output_dtype=torch.float32)
    err = (tensor - x_hat).abs().max().item()
    # Half a bin in the worst case.
    bin_size = (tensor.abs().amax() / q._qmax).item()
    assert err <= bin_size + 1e-3


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_int_asymmetric_roundtrip(tensor: torch.Tensor, bits: int) -> None:
    q = IntQuantizer(bits=bits, symmetric=False, per_channel=True)
    packed, scale, zp = q.quantize(tensor)
    x_hat = q.dequantize(packed, scale, zp, output_dtype=torch.float32)
    err = (tensor - x_hat).abs().max().item()
    rng = (tensor.amax() - tensor.amin()).item()
    bin_size = rng / (q._qmax - q._qmin)
    assert err <= bin_size + 1e-3


def test_int_per_tensor() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    q = IntQuantizer(bits=8, symmetric=True, per_channel=False)
    packed, scale, zp = q.quantize(x)
    assert scale.dim() == 0
    assert zp.dim() == 0
    x_hat = q.dequantize(packed, scale, zp, output_dtype=torch.float32)
    assert x_hat.shape == x.shape


def test_int_per_group() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 8)
    q = IntQuantizer(bits=4, symmetric=True, per_channel=False, group_size=4)
    packed, scale, zp = q.quantize(x)
    # group_size=4 → 2 groups per row → scale shape (2, 2).
    assert scale.shape == (2, 2)
    x_hat = q.dequantize(packed, scale, zp, output_dtype=torch.float32)
    assert x_hat.shape == x.shape
    err = (x - x_hat).abs().max().item()
    assert err < 0.5


def test_packing_inverse_unpacking() -> None:
    torch.manual_seed(0)
    for bits in (2, 4, 8):
        q = IntQuantizer(bits=bits, symmetric=True, per_channel=True)
        x = torch.randn(4, 16) * 2
        packed, scale, zp = q.quantize(x)
        x_hat = q.dequantize(packed, scale, zp, output_dtype=torch.float32)
        assert x_hat.shape == x.shape


def test_dispatch_int8() -> None:
    q = get_quantizer("int8")
    assert q.name == "int8"


def test_dispatch_int4() -> None:
    q = get_quantizer("int4", symmetric=False)
    assert q.symmetric is False


def test_dispatch_int2() -> None:
    q = get_quantizer("int2")
    packed, scale, zp = q.quantize(torch.randn(2, 4))
    assert packed.dtype == torch.uint8


def test_dispatch_fp16() -> None:
    q = get_quantizer("fp16")
    x = torch.randn(4, 8)
    packed, scale, zp = q.quantize(x)
    assert packed.dtype == torch.float16


def test_dispatch_bf16() -> None:
    q = get_quantizer("bf16")
    x = torch.randn(4, 8)
    packed, scale, zp = q.quantize(x)
    assert packed.dtype == torch.bfloat16


def test_quantize_dequantize_roundtrip() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 12)
    payload = quantize_tensor(x, dtype="int4")
    x_hat = dequantize_tensor(payload, dtype="int4")
    assert x_hat.shape == x.shape


def test_estimate_int_bytes() -> None:
    assert estimate_int_bytes(8, 8) == 8
    assert estimate_int_bytes(8, 4) == 4
    assert estimate_int_bytes(8, 2) == 2


def test_invalid_bits() -> None:
    with pytest.raises(ValueError, match="bits"):
        IntQuantizer(bits=3)


def test_int_group_size_not_divisible() -> None:
    x = torch.randn(2, 7)  # 7 not divisible by 4
    q = IntQuantizer(bits=4, group_size=4, per_channel=False)
    with pytest.raises(AssertionError):
        q.quantize(x)


def test_int2_shape_after_packing() -> None:
    """Bit 2 packing: 4 entries per byte."""
    torch.manual_seed(0)
    x = torch.randn(2, 8) * 0.5
    q = IntQuantizer(bits=2, symmetric=True, per_channel=True)
    packed, scale, zp = q.quantize(x)
    # 8 entries per row, packed at 2 bits → 2 bytes per row, 4 rows = 8 bytes.
    assert packed.dtype == torch.uint8


def test_int4_shape_after_packing() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 8)
    q = IntQuantizer(bits=4, symmetric=True, per_channel=True)
    packed, scale, zp = q.quantize(x)
    # 8 entries per row, 4 bits → 4 bytes per row.
    assert packed.dtype == torch.uint8