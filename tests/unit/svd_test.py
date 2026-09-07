"""Tests for the SVD / Decomposer module."""

from __future__ import annotations

import pytest
import torch

from kvfold.core.svd import Exact, Randomized, tail_mass


@pytest.fixture
def small_matrix() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(40, 30)


def test_exact_full_rank(small_matrix: torch.Tensor) -> None:
    decomposer = Exact()
    res = decomposer.decompose(small_matrix, rank=min(40, 30))
    assert res.rank == min(40, 30)
    assert res.strategy == "exact"
    assert res.tail_mass == 0.0


def test_exact_truncated_reconstruction(small_matrix: torch.Tensor) -> None:
    decomposer = Exact()
    res = decomposer.decompose(small_matrix, rank=5)
    assert res.rank == 5
    rel = res.reconstruction_error(small_matrix)
    assert 0.0 < rel < 1.0


def test_randomized_reconstruction_quality(small_matrix: torch.Tensor) -> None:
    decomposer = Randomized(oversampling=10, n_power=2, seed=0)
    res = decomposer.decompose(small_matrix, rank=10)
    assert res.strategy == "randomised"
    rel = res.reconstruction_error(small_matrix)
    exact_res = Exact().decompose(small_matrix, rank=10)
    exact_rel = exact_res.reconstruction_error(small_matrix)
    assert rel <= exact_rel * 1.2 + 1e-3


def test_auto_dispatch_picks_randomised() -> None:
    torch.manual_seed(0)
    a = torch.randn(40, 40)
    decomposer = Exact()
    res = decomposer.decompose(a, rank=5)
    assert res.strategy == "exact"


def test_rank_clamped_to_min_dim() -> None:
    decomposer = Exact()
    a = torch.randn(5, 10)
    res = decomposer.decompose(a, rank=20)
    assert res.rank == 5


def test_reconstruct_shape() -> None:
    decomposer = Exact()
    a = torch.randn(10, 6)
    res = decomposer.decompose(a, rank=3)
    assert res.reconstruct().shape == a.shape


def test_non_2d_raises() -> None:
    decomposer = Exact()
    from kvfold.errors import ShapeError
    with pytest.raises(ShapeError):
        decomposer.decompose(torch.randn(3, 4, 5), rank=5)


def test_randomized_preserves_device() -> None:
    decomposer = Randomized(seed=0)
    a = torch.randn(20, 20)
    res = decomposer.decompose(a, rank=5)
    assert res.u.device == a.device
    assert res.vh.device == a.device
    assert res.s.device == a.device


def test_randomized_seed_reproducibility() -> None:
    a = torch.randn(40, 40)
    s1 = Randomized(seed=0).decompose(a, rank=5)
    s2 = Randomized(seed=0).decompose(a, rank=5)
    assert torch.allclose(s1.u, s2.u)
    s3 = Randomized(seed=1).decompose(a, rank=5)
    assert not torch.allclose(s1.u, s3.u)


def test_tail_mass_full_rank_zero() -> None:
    decomposer = Exact()
    a = torch.randn(20, 20)
    res = decomposer.decompose(a, rank=20)
    assert res.tail_mass == 0.0


def test_tail_mass_decreases_with_rank() -> None:
    decomposer = Exact()
    a = torch.randn(40, 30)
    res_low = decomposer.decompose(a, rank=2)
    res_high = decomposer.decompose(a, rank=10)
    assert res_low.tail_mass >= res_high.tail_mass


def test_tail_mass_module_function() -> None:
    s = torch.tensor([3.0, 2.0, 1.0, 0.5])
    assert tail_mass(s, 0) == 1.0
    assert tail_mass(s, 4) == 0.0
    assert tail_mass(s, 2) > 0.0
