"""Tests for the Tucker / ST-HOSVD module."""

from __future__ import annotations

import pytest
import torch

from kvcompress.compressor.svd import SVD
from kvcompress.compressor.tucker import (
    mode_n_fold,
    mode_n_unfold,
    partial_tucker_st_hosvd,
    reconstruct_partial_tucker,
)


@pytest.fixture
def tensor() -> torch.Tensor:
    torch.manual_seed(0)
    # Small but not degenerate.
    return torch.randn(4, 16, 8)


def test_mode_unfold_roundtrip(tensor: torch.Tensor) -> None:
    for mode in range(3):
        unfolded = mode_n_unfold(tensor, mode)
        assert unfolded.shape[0] == tensor.shape[mode]
        assert unfolded.numel() == tensor.numel()
        folded = mode_n_fold(unfolded, mode, tensor.shape)
        assert torch.allclose(folded, tensor)


def test_partial_tucker_shapes(tensor: torch.Tensor) -> None:
    factors = partial_tucker_st_hosvd(tensor, r_token=4, r_feature=4)
    m, t, d = tensor.shape
    assert factors.core.shape == (m, 4, 4)
    assert factors.u_token.shape == (t, 4)
    assert factors.u_feature.shape == (d, 4)


def test_partial_tucker_reconstruction_improves_with_rank(
    tensor: torch.Tensor,
) -> None:
    f_low = partial_tucker_st_hosvd(tensor, r_token=1, r_feature=1)
    f_high = partial_tucker_st_hosvd(tensor, r_token=4, r_feature=4)
    x_low = reconstruct_partial_tucker(f_low, tensor.shape)
    x_high = reconstruct_partial_tucker(f_high, tensor.shape)
    err_low = torch.linalg.norm(tensor - x_low) / torch.linalg.norm(tensor)
    err_high = torch.linalg.norm(tensor - x_high) / torch.linalg.norm(tensor)
    assert err_high < err_low


def test_full_rank_is_identity(tensor: torch.Tensor) -> None:
    m, t, d = tensor.shape
    factors = partial_tucker_st_hosvd(tensor, r_token=t, r_feature=d)
    x_hat = reconstruct_partial_tucker(factors, tensor.shape)
    err = torch.linalg.norm(tensor - x_hat) / torch.linalg.norm(tensor)
    # Numerical noise from matmul.
    assert err < 1e-4


def testtail_mass_zero_at_full_rank(tensor: torch.Tensor) -> None:
    m, t, d = tensor.shape
    factors = partial_tucker_st_hosvd(tensor, r_token=t, r_feature=d)
    assert factors.token_tail_mass == 0.0
    assert factors.feature_tail_mass == 0.0


def test_token_rank_caps_at_t(tensor: torch.Tensor) -> None:
    factors = partial_tucker_st_hosvd(tensor, r_token=999, r_feature=4)
    assert factors.r_token <= tensor.shape[1]


def test_feature_rank_caps_at_d(tensor: torch.Tensor) -> None:
    factors = partial_tucker_st_hosvd(tensor, r_token=4, r_feature=999)
    assert factors.r_feature <= tensor.shape[2]


def test_shared_svd_seed() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 8, 6)
    a = partial_tucker_st_hosvd(x, r_token=3, r_feature=3, svd=SVD(seed=1))
    b = partial_tucker_st_hosvd(x, r_token=3, r_feature=3, svd=SVD(seed=1))
    assert torch.allclose(a.u_token, b.u_token)
    assert torch.allclose(a.u_feature, b.u_feature)


def test_non_3d_raises() -> None:
    with pytest.raises(ValueError, match="3-D"):
        partial_tucker_st_hosvd(torch.randn(4, 4))


def test_orthonormal_columns() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 12, 8)
    factors = partial_tucker_st_hosvd(x, r_token=4, r_feature=4)
    # U_T columns are orthonormal: U^T U = I_r.
    utg = factors.u_token.t() @ factors.u_token
    assert torch.allclose(utg, torch.eye(4), atol=1e-4)
    udg = factors.u_feature.t() @ factors.u_feature
    assert torch.allclose(udg, torch.eye(4), atol=1e-4)
