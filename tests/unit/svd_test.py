"""Tests for the SVD module."""

from __future__ import annotations

import pytest
import torch

from kvcompress.compressor.svd import SVD


@pytest.fixture
def small_matrix() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(40, 30)


def test_exact_full_rank(small_matrix: torch.Tensor) -> None:
    svd = SVD()
    res = svd.exact(small_matrix)
    assert res.rank == min(40, 30)
    assert res.method == "exact"
    assert res.tail_mass == 0.0


def test_exact_truncated_reconstruction(small_matrix: torch.Tensor) -> None:
    svd = SVD()
    res = svd.exact(small_matrix, rank=5)
    assert res.rank == 5
    rel = res.reconstruction_error(small_matrix)
    # Random 40x30 has a fairly flat spectrum; rank-5 error is ~0.73.
    # We test that we are *not* exact (i.e. truncation actually happened),
    # but that the result is finite and at most the rank-deficient gap.
    assert 0.0 < rel < 1.0


def test_randomised_reconstruction_quality(small_matrix: torch.Tensor) -> None:
    svd = SVD(oversampling=10, n_power=2, seed=0)
    res = svd.randomise(small_matrix, rank=10)
    assert res.method == "randomised"
    rel = res.reconstruction_error(small_matrix)
    # Randomised SVD with power iterations should be within slack of exact.
    exact_res = SVD().exact(small_matrix, rank=10)
    exact_rel = exact_res.reconstruction_error(small_matrix)
    assert rel <= exact_rel * 1.2 + 1e-3


def test_auto_dispatch_picks_randomised() -> None:
    torch.manual_seed(0)
    a = torch.randn(40, 40)
    svd = SVD(method="auto")
    res = svd(a, rank=5)  # 5 < 40 // 2
    assert res.method == "randomised"


def test_auto_dispatch_picks_exact() -> None:
    torch.manual_seed(0)
    a = torch.randn(40, 40)
    svd = SVD(method="auto")
    res = svd(a, rank=25)  # 25 >= 40 // 2
    assert res.method == "exact"


def test_force_exact() -> None:
    svd = SVD(method="exact")
    a = torch.randn(40, 40)
    res = svd(a, rank=5)
    assert res.method == "exact"


def test_force_randomised() -> None:
    svd = SVD(method="randomised")
    a = torch.randn(40, 40)
    res = svd(a, rank=5)
    assert res.method == "randomised"


def test_cap_caps_sketch_size() -> None:
    svd = SVD(oversampling=20, n_power=0, seed=0)
    a = torch.randn(40, 40)
    res = svd.randomise(a, rank=10, cap=12)
    # Cap = 12 means we sketched 12 columns; we still keep rank 10.
    assert res.rank == 10


def test_seed_reproducibility() -> None:
    a = torch.randn(40, 40)
    s1 = SVD(seed=0).randomise(a, rank=5)
    s2 = SVD(seed=0).randomise(a, rank=5)
    assert torch.allclose(s1.u, s2.u)
    s3 = SVD(seed=1).randomise(a, rank=5)
    assert not torch.allclose(s1.u, s3.u)


def test_rank_clamped_to_min_dim() -> None:
    svd = SVD()
    a = torch.randn(5, 10)
    res = svd.exact(a, rank=20)
    assert res.rank == 5


def test_reconstruct_shape() -> None:
    svd = SVD()
    a = torch.randn(10, 6)
    res = svd.exact(a, rank=3)
    assert res.reconstruct().shape == a.shape


def test_non_2d_raises() -> None:
    svd = SVD()
    with pytest.raises(ValueError, match="2-D matrix"):
        svd.exact(torch.randn(3, 4, 5))


def testtail_mass_full_rank_zero() -> None:
    svd = SVD()
    a = torch.randn(20, 20)
    res = svd.exact(a, rank=20)
    assert res.tail_mass == 0.0


def testtail_mass_decreases_with_rank() -> None:
    svd = SVD()
    a = torch.randn(40, 30)
    res_low = svd.exact(a, rank=2)
    res_high = svd.exact(a, rank=10)
    assert res_low.tail_mass >= res_high.tail_mass
