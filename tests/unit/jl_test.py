"""Tests for the JL projection module."""

from __future__ import annotations

import pytest
import torch

from kvcompress.compressor.jl import (
    cached_projection,
    clear_projection_cache,
    gaussian_projection,
    rademacher_projection,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_projection_cache()


def test_gaussian_projection_shape() -> None:
    p = gaussian_projection(8, 16, seed=0)
    assert p.matrix.shape == (8, 16)
    assert p.distribution == "gaussian"


def test_rademacher_projection_shape() -> None:
    p = rademacher_projection(8, 16, seed=0)
    assert p.matrix.shape == (8, 16)
    assert p.distribution == "rademacher"


def test_seed_reproducibility() -> None:
    p1 = gaussian_projection(4, 4, seed=42)
    p2 = gaussian_projection(4, 4, seed=42)
    assert torch.allclose(p1.matrix, p2.matrix)
    p3 = gaussian_projection(4, 4, seed=43)
    assert not torch.allclose(p1.matrix, p3.matrix)


def test_apply_last_axis() -> None:
    p = gaussian_projection(4, 4, seed=0)
    x = torch.randn(3, 5, 4)
    y = p.apply(x)
    assert y.shape == (3, 5, 4)


def test_apply_inverse_roundtrip_shape() -> None:
    p = gaussian_projection(4, 4, seed=0)
    x = torch.randn(2, 3, 4)
    y = p.apply(x)
    z = p.apply_inverse(y)
    assert z.shape == x.shape


def test_norm_preservation_gaussian() -> None:
    """JL should preserve squared norm on average (within slack)."""
    torch.manual_seed(0)
    p = gaussian_projection(64, 64, seed=0)
    x = torch.randn(1024, 64)
    y = p.apply(x)
    ratio = (y * y).sum(dim=-1) / (x * x).sum(dim=-1)
    # Mean ratio should be ~1.0; allow wide slack for finite samples.
    assert abs(ratio.mean().item() - 1.0) < 0.1


def test_cached_projection_returns_same_object() -> None:
    p1 = cached_projection(8, 16, seed=0)
    p2 = cached_projection(8, 16, seed=0)
    assert p1 is p2


def test_cached_projection_distribution_switch() -> None:
    p_g = cached_projection(4, 4, distribution="gaussian", seed=0)
    p_r = cached_projection(4, 4, distribution="rademacher", seed=0)
    assert p_g.distribution == "gaussian"
    assert p_r.distribution == "rademacher"
    assert not torch.allclose(p_g.matrix, p_r.matrix)


def test_rademacher_sparsity() -> None:
    p = rademacher_projection(16, 16, seed=0, sparsity=0.25)
    nz = (p.matrix != 0).float().mean().item()
    # Expect ~0.25, with some slack.
    assert 0.18 < nz < 0.32


def test_apply_shape_mismatch_raises() -> None:
    p = gaussian_projection(4, 4, seed=0)
    with pytest.raises(ValueError, match="trailing dim mismatch"):
        p.apply(torch.randn(2, 3, 5))