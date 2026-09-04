"""Regression tests for the JL projection cache dtype Tier 0 bug (B-06)."""

from __future__ import annotations

import torch

from kvfold.core.jl import CACHE


def test_dtype_in_cache_key() -> None:
    """Two projections of the same (shape, seed) but different dtypes must NOT collide."""
    CACHE.clear()
    try:
        p_f32 = CACHE.get_or_build(out_dim=32, in_dim=32, distribution="gaussian", seed=0, device="cpu", dtype=torch.float32)
        p_f16 = CACHE.get_or_build(out_dim=32, in_dim=32, distribution="gaussian", seed=0, device="cpu", dtype=torch.float16)
        assert p_f32.dtype == torch.float32
        assert p_f16.dtype == torch.float16
        info = CACHE.info()
        assert info["size"] >= 2
    finally:
        CACHE.clear()
