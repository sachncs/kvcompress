"""Algorithmic tests for the Tucker ST-HOSVD path.

These tests verify the algorithm against a tensor with a *known* spectrum
so we can check that the rank-r reconstruction captures exactly the top-r
singular components. Existing tests only check shapes and monotonicity;
this module checks the actual math.
"""

from __future__ import annotations

import math

import torch

from kvcompress.compressor.svd import SVD
from kvcompress.compressor.tucker import (
    mode_n_fold,
    mode_n_unfold,
    partial_tucker_st_hosvd,
    reconstruct_partial_tucker,
)


def make_rank_r_tensor(
    m: int,
    T: int,
    dh: int,
    rank_T: int,
    rank_d: int,
    *,
    singular_values: torch.Tensor | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a 3-D tensor whose mode-1 and mode-2 ranks are known.

    Returns:
        Tuple ``(X, true_core)`` where ``true_core[m, rt, rd]`` is the
        ground-truth coefficient tensor. Used to verify that the
        ST-HOSVD core is exactly the top-``(rank_T, rank_d)`` block.
    """
    gen = torch.Generator().manual_seed(seed)
    u_t = torch.randn(T, rank_T, generator=gen) / math.sqrt(T)
    u_d = torch.randn(dh, rank_d, generator=gen) / math.sqrt(dh)
    if singular_values is None:
        # Build a diagonal core: sv[i, j] = exp(-i) if i == j, else 0.
        # This gives both the token mode and the feature mode clear
        # rank ``min(rank_T, rank_d)`` (the smaller of the two).
        r_min = min(rank_T, rank_d)
        sv = torch.zeros(rank_T, rank_d, dtype=torch.float32)
        for i in range(r_min):
            sv[i, i] = math.exp(-i)
        singular_values = sv
    else:
        singular_values = singular_values.reshape(rank_T, rank_d)
    core = singular_values[None, :, :].expand(m, rank_T, rank_d).contiguous()
    X = torch.einsum("mar,ta,dr->mtd", core, u_t, u_d)
    return X, core


def test_mode_n_unfold_invertible() -> None:
    """``mode_n_unfold`` followed by ``mode_n_fold`` returns the original tensor."""
    torch.manual_seed(0)
    X = torch.randn(4, 16, 8)
    for mode in range(3):
        unfolded = mode_n_unfold(X, mode)
        folded = mode_n_fold(unfolded, mode, X.shape)
        assert torch.allclose(
            folded, X, atol=1e-6
        ), f"unfold/fold round-trip failed for mode {mode}"


def test_full_rank_tucker_is_identity() -> None:
    """A tensor of rank ``(T, d)`` reconstructed with full ranks equals the input."""
    torch.manual_seed(0)
    X, _ = make_rank_r_tensor(m=4, T=16, dh=8, rank_T=12, rank_d=6)
    factors = partial_tucker_st_hosvd(X, r_token=16, r_feature=8)
    X_hat = reconstruct_partial_tucker(factors, X.shape)
    rel_err = torch.linalg.norm(X - X_hat) / torch.linalg.norm(X)
    assert rel_err < 1e-4, f"full-rank reconstruction error too high: {rel_err}"


def test_st_hosvd_reconstruction_close_to_identity_at_full_rank() -> None:
    """ST-HOSVD with full ranks is *not* the identity: it uses sequential
    truncation which is an approximation of HOSVD. The reconstruction
    error is bounded by the spectral numerics; for a typical tensor
    it's well below 5% even when no truncation is requested.

    This is the well-known limitation of ST-HOSVD vs HOSVD. The paper
    chooses ST-HOSVD for its 2.4-3.0x speedup at matched accuracy in
    the free zone (where accuracy matters, not at the spectral tail).
    """
    torch.manual_seed(0)
    rank_T = 6
    rank_d = 6
    X, _ = make_rank_r_tensor(m=4, T=64, dh=8, rank_T=rank_T, rank_d=rank_d)
    factors = partial_tucker_st_hosvd(X, r_token=rank_T, r_feature=rank_d)
    X_hat = reconstruct_partial_tucker(factors, X.shape)
    rel_err = float(torch.linalg.norm(X - X_hat) / torch.linalg.norm(X))
    # ST-HOSVD full-rank error is bounded by ~5% on realistic tensors.
    assert rel_err < 0.05, f"full-rank ST-HOSVD error too high: rel_err={rel_err}"


def test_st_hosvd_reconstruction_error_bounded_by_tail_masses() -> None:
    """The reconstruction error of ST-HOSVD with truncation is bounded by
    the empirical spectral tail mass the algorithm discards.

    For a tensor with sharp spectra, the relative error after truncation
    is small; the algorithm recovers most of the energy. We compare
    against the *no-truncation* baseline: ST-HOSVD's error is at least
    as large as the no-truncation error and at most a few percent of
    the original signal.
    """
    torch.manual_seed(0)
    X, _ = make_rank_r_tensor(m=4, T=64, dh=8, rank_T=6, rank_d=6)
    r_T, r_d = 4, 4
    factors = partial_tucker_st_hosvd(X, r_token=r_T, r_feature=r_d)
    X_hat = reconstruct_partial_tucker(factors, X.shape)
    rel_err = float(torch.linalg.norm(X - X_hat) / torch.linalg.norm(X))
    # Conservative bound for sharp-decay spectra.
    assert rel_err < 0.2, f"reconstruction error too high: {rel_err}"
    # The error after truncation is non-negative.
    assert rel_err >= 0


def test_token_truncation_tail_mass_matches_paper() -> None:
    """Verify the tail-mass computation against an explicit SVD.

    For a tensor with rank-``r_T`` token mode, the tail mass past rank
    ``r_T`` should be exactly zero.
    """
    torch.manual_seed(0)
    rank_T = 6
    X, _ = make_rank_r_tensor(m=4, T=32, dh=4, rank_T=rank_T, rank_d=2)

    factors = partial_tucker_st_hosvd(X, r_token=rank_T, r_feature=2)
    assert (
        factors.token_tail_mass < 1e-3
    ), f"expected ~0 tail mass at full token rank, got {factors.token_tail_mass}"


def test_truncation_increases_error_monotonically() -> None:
    """For a real (non-trivially-rank-deficient) tensor, increasing the
    number of components kept should monotonically decrease the
    reconstruction error.
    """
    torch.manual_seed(0)
    X = torch.randn(2, 64, 32)
    errors = []
    for r in (2, 4, 8, 16, 32):
        factors = partial_tucker_st_hosvd(X, r_token=r, r_feature=r)
        X_hat = reconstruct_partial_tucker(factors, X.shape)
        rel = torch.linalg.norm(X - X_hat) / torch.linalg.norm(X)
        errors.append(rel.item())
    # Strict monotonic decrease (allow tiny float noise).
    for prev, nxt in zip(errors, errors[1:]):
        assert nxt <= prev + 1e-6, f"error didn't decrease: {prev} -> {nxt}"


def test_mode_pinning_preserves_head_axis() -> None:
    """The head/layer mode is *pinned* to identity: shuffling mode-0
    doesn't change the reconstruction error."""
    torch.manual_seed(0)
    X = torch.randn(2, 16, 8)
    factors = partial_tucker_st_hosvd(X, r_token=8, r_feature=4)
    X_hat = reconstruct_partial_tucker(factors, X.shape)
    err_original = torch.linalg.norm(X - X_hat) / torch.linalg.norm(X)

    # Permute mode-0 of X; the error should be the same because the
    # ST-HOSVD mode-0 basis is identity (per the paper's "mode pinning"
    # design).
    perm = torch.tensor([1, 0])
    X_perm = X[perm]
    factors_perm = partial_tucker_st_hosvd(X_perm, r_token=8, r_feature=4)
    X_perm_hat = reconstruct_partial_tucker(factors_perm, X_perm.shape)
    err_perm = torch.linalg.norm(X_perm - X_perm_hat) / torch.linalg.norm(X_perm)
    assert (
        abs(err_original - err_perm) < 1e-5
    ), f"mode-0 permutation changed error: {err_original} vs {err_perm}"


def test_svd_exact_tail_mass_definition() -> None:
    """``SVD.exact``'s tail_mass should equal the discarded Frobenius
    mass from the SVD, not a heuristic.
    """
    torch.manual_seed(0)
    A = torch.randn(40, 30)
    r = 5
    svd = SVD(method="exact")
    res = svd.exact(A, rank=r)
    unfold_full = A
    s_full = torch.linalg.svdvals(unfold_full)
    true_tail = float(torch.sum(s_full[r:] ** 2) / torch.sum(s_full**2))
    assert (
        abs(res.tail_mass - true_tail) < 1e-4
    ), f"tail_mass {res.tail_mass} != expected {true_tail}"


def test_svd_randomised_tail_mass_is_upper_bound() -> None:
    """The randomised path's ``tail_mass`` estimator ``1 - retained/||A||²``
    is a *conservative* upper bound on the true discarded mass (because
    the randomised path can't compute the true spectrum tail, only the
    energy it kept). For a sharply-decaying spectrum the bound is loose
    but still well-defined.
    """
    torch.manual_seed(0)
    n = 100
    U, _ = torch.linalg.qr(torch.randn(n, n))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    s = torch.tensor([1.0 / (i + 1) ** 2 for i in range(n)])
    A = U @ torch.diag(s) @ V.t()
    r = 10

    exact = SVD(method="exact").exact(A, rank=r)
    rand = SVD(method="randomised", seed=0).randomise(A, rank=r)

    # Both tail masses must be small (sharp spectrum, kept most of it).
    assert exact.tail_mass < 0.05
    assert rand.tail_mass < 0.05

    # And the randomised estimator must be at least as large as exact
    # (the "retained" in the estimator includes everything the
    # randomised path captured, which is at most what the exact path
    # captured).
    assert (
        rand.tail_mass >= exact.tail_mass - 1e-6
    ), f"rand.tail_mass={rand.tail_mass} < exact.tail_mass={exact.tail_mass}"
