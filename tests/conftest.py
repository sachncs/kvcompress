"""Shared pytest fixtures for the kvfold test suite.

Fixtures:

- ``tiny_kv``      — (heads=8,  length=64,  head_dim=32) random K/V
- ``medium_kv``    — (heads=8,  length=256, head_dim=64) random K/V
- ``large_kv``     — (heads=8,  length=1024, head_dim=128) random K/V
- ``kv_factory``   — parametric factory returning (K, V) at chosen shape
- ``deterministic_seed`` — pins torch.manual_seed for the test
- ``cache_factory``      — builds a Cache with a stub compressor
- ``compressor_factory`` — builds a Compressor for a given method name
"""

from __future__ import annotations

from typing import Callable

import pytest
import torch

from kvfold.api import build_compressor


@pytest.fixture
def deterministic_seed() -> int:
    torch.manual_seed(0)
    return 0


@pytest.fixture
def tiny_kv(deterministic_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(8, 64, 32)
    v = torch.randn(8, 64, 32)
    return k, v


@pytest.fixture
def medium_kv(deterministic_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(8, 256, 64)
    v = torch.randn(8, 256, 64)
    return k, v


@pytest.fixture
def large_kv(deterministic_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(8, 1024, 128)
    v = torch.randn(8, 1024, 128)
    return k, v


@pytest.fixture
def kv_factory(deterministic_seed: int) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    def make(heads: int = 8, length: int = 256, head_dim: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.randn(heads, length, head_dim)
        v = torch.randn(heads, length, head_dim)
        return k, v

    return make


@pytest.fixture
def compressor_factory() -> Callable[..., object]:
    def make(method: str = "jolt", **kwargs: object) -> object:
        return build_compressor(method, **kwargs)

    return make
