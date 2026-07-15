"""Tests for Triton kernels (no-op fallback path)."""

from __future__ import annotations

import pytest
import torch

from kvcompress.kernels.triton.compression import (
    is_triton_available,
    jl_project,
    quantize_int8,
    tucker_reconstruct,
)
from kvcompress.kernels.triton.tucker_reconstruct import triton_tucker_reconstruct


def test_is_triton_available() -> None:
    """Just check that the import-time check works without raising."""
    assert isinstance(is_triton_available(), bool)


def test_tucker_reconstruct_fallback() -> None:
    torch.manual_seed(0)
    core = torch.randn(4, 8, 4)
    u_token = torch.randn(16, 8)
    u_feature = torch.randn(8, 4)
    expected = torch.einsum("mar,ta,dr->mtd", core, u_token, u_feature)
    out = tucker_reconstruct(core, u_token, u_feature)
    assert torch.allclose(out, expected)


def test_jl_project_fallback() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    matrix = torch.randn(16, 16)
    expected = x @ matrix.t()
    out = jl_project(x, matrix)
    assert torch.allclose(out, expected)


def test_quantize_int8_fallback() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    packed, scale, zp = quantize_int8(x)
    assert packed.dtype == torch.uint8


def test_triton_tucker_reconstruct_fallback() -> None:
    """When Triton isn't available, this should fall back to PyTorch."""
    if is_triton_available():
        pytest.skip("Triton is available; would test the JIT path")
    torch.manual_seed(0)
    core = torch.randn(2, 4, 4)
    u_token = torch.randn(8, 4)
    u_feature = torch.randn(8, 4)
    expected = torch.einsum("mar,ta,dr->mtd", core, u_token, u_feature)
    out = triton_tucker_reconstruct(core, u_token, u_feature)
    assert torch.allclose(out, expected)


def test_vllm_adapter_import_succeeds_without_vllm() -> None:
    """vllm adapter imports even when vllm is not installed."""
    from kvcompress.adapters.vllm import is_vllm_available

    # Either True or False; just check it doesn't raise.
    assert isinstance(is_vllm_available(), bool)