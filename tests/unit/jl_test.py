"""Tests for the JL projection module."""

from __future__ import annotations

import pytest
import torch

from kvfold.core.jl import CACHE, Gaussian, Rademacher, Sparse


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    CACHE.clear()


def gaussian_projection(out_dim: int, in_dim: int, *, seed: int, **kwargs: object):
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("dtype", torch.float32)
    return CACHE.get_or_build(out_dim, in_dim, distribution="gaussian", seed=seed, **kwargs)


def rademacher_projection(out_dim: int, in_dim: int, *, seed: int, **kwargs: object):
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("dtype", torch.float32)
    return CACHE.get_or_build(out_dim, in_dim, distribution="rademacher", seed=seed, **kwargs)


def sparse_projection(out_dim: int, in_dim: int, *, seed: int, sparsity: float = 0.1):
    return CACHE.get_or_build(out_dim, in_dim, distribution="sparse", seed=seed, device="cpu", dtype=torch.float32, projector=Sparse(sparsity))


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
    torch.manual_seed(0)
    p = gaussian_projection(64, 64, seed=0)
    x = torch.randn(1024, 64)
    y = p.apply(x)
    ratio = (y * y).sum(dim=-1) / (x * x).sum(dim=-1)
    assert abs(ratio.mean().item() - 1.0) < 0.1


def test_cache_returns_same_object() -> None:
    p1 = gaussian_projection(8, 16, seed=0)
    p2 = gaussian_projection(8, 16, seed=0)
    assert p1 is p2


def test_cache_distribution_switch() -> None:
    p_g = gaussian_projection(4, 4, seed=0)
    p_r = rademacher_projection(4, 4, seed=0)
    assert p_g.distribution == "gaussian"
    assert p_r.distribution == "rademacher"
    assert not torch.allclose(p_g.matrix, p_r.matrix)


def test_rademacher_entries() -> None:
    p = rademacher_projection(16, 16, seed=0)
    unique = torch.unique(p.matrix)
    rounded = sorted(round(float(u), 4) for u in unique)
    assert rounded == [-1.0, 1.0] or len(rounded) <= 2


def test_apply_shape_mismatch_raises() -> None:
    p = gaussian_projection(4, 4, seed=0)
    from kvfold.errors import ShapeError
    with pytest.raises(ShapeError):
        p.apply(torch.randn(2, 3, 5))
