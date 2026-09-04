"""Tests for the LowRank and IntQuantOnly baseline compressors."""

from __future__ import annotations

import pytest
import torch

from kvfold.core.lowrank import Low
from kvfold.core.quantization_only import IntQuant


@pytest.fixture
def kv() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(4, 32, 16), torch.randn(4, 32, 16)


def test_lowrank_roundtrip(kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    K, V = kv
    comp = Low(rank=4)
    kp, vp = comp.compress(K, V)
    k_hat, v_hat = comp.decompress(kp, vp)
    assert k_hat.shape == K.shape
    assert v_hat.shape == V.shape


def test_lowrank_bytes_smaller(kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    K, V = kv
    comp = Low(rank=4)
    kp, vp = comp.compress(K, V)
    original = K.numel() * K.element_size() * 2
    assert kp.bytes_compressed + vp.bytes_compressed < original


def test_int_quant_roundtrip(kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    K, V = kv
    comp = IntQuant(bits=4)
    kp, vp = comp.compress(K, V)
    k_hat, v_hat = comp.decompress(kp, vp)
    assert k_hat.shape == K.shape


def test_int_quant_int8_roundtrip(kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    K, V = kv
    comp = IntQuant(bits=8)
    kp, vp = comp.compress(K, V)
    k_hat, v_hat = comp.decompress(kp, vp)
    assert k_hat.shape == K.shape


def test_int_quant_int2_roundtrip(kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    K, V = kv
    comp = IntQuant(bits=2)
    kp, vp = comp.compress(K, V)
    k_hat, v_hat = comp.decompress(kp, vp)
    assert k_hat.shape == K.shape


def test_int_quant_accepts_arbitrary_shape() -> None:
    """IntQuantOnly works on any shape since it operates on the last dim."""
    comp = IntQuant(bits=4)
    K = torch.randn(2, 4)
    V = torch.randn(2, 4)
    kp, vp = comp.compress(K, V)
    assert kp.shape == (2, 4)
    assert vp.shape == (2, 4)


def test_lowrank_shape_mismatch() -> None:
    comp = Low(rank=4)
    K = torch.randn(4, 32, 16)
    V = torch.randn(4, 16, 16)
    with pytest.raises(ValueError):
        comp.compress(K, V)
