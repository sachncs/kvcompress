"""Partial Tucker decomposition via sequentially truncated HOSVD (ST-HOSVD).

The KV cache at one layer is a third-order tensor
``X ∈ R^{m × T × dh}`` where ``m`` merges head count and number of layers in
the group, ``T`` is the number of tokens, and ``dh`` is the per-head feature
dim. JoLT applies *partial* Tucker: identity factors on modes 0 (head/layer
merged) and 1 (tokens), with rank truncation only on modes 1 and 2.

Mode pinning is the paper's empirical finding (Appendix B.2): a full
four-mode allocator reproduces the partial method byte-for-byte by driving
the head and layer ranks back to their full size, and the partial form is
2.4-3.0× faster than full HOOI at identical error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from kvcompress.compressor.svd import SVD

log = logging.getLogger(__name__)


@dataclass
class TuckerFactors:
    """Output of :func:`partial_tucker_st_hosvd`.

    The reconstruction is::

        X̂ = core ×_1 I_m ×_2 U_T ×_3 U_d
          = einsum("mtr,md,tn,dr->mtd", core, eye_m, U_T, U_d)

    With identity on mode 1 this collapses to a 2-D matmul sequence.

    Attributes:
        core: shape ``(m, rT, rd)`` — the compressed core.
        u_token: shape ``(T, rT)`` — token-mode basis (orthonormal columns).
        u_feature: shape ``(dh, rd)`` — feature-mode basis.
        token_sv: singular values of the token-mode unfolding (length T).
        feature_sv: singular values of the feature-mode unfolding (length dh).
        token_tail_mass: relative Frobenius mass discarded by truncating
            token-mode to rank ``rT``.
        feature_tail_mass: same for the feature-mode at rank ``rd``.
    """

    core: torch.Tensor
    u_token: torch.Tensor
    u_feature: torch.Tensor
    token_sv: torch.Tensor
    feature_sv: torch.Tensor
    token_tail_mass: float
    feature_tail_mass: float

    @property
    def r_token(self) -> int:
        return int(self.u_token.shape[1])

    @property
    def r_feature(self) -> int:
        return int(self.u_feature.shape[1])

    @property
    def bytes_factors(self) -> int:
        # Stored as fp16 by convention.
        elem = 2  # fp16
        return int(self.core.numel() + self.u_token.numel() + self.u_feature.numel()) * elem


def mode_n_unfold(x: torch.Tensor, mode: int) -> torch.Tensor:
    """Mode-``mode`` unfolding of a 3-D tensor.

    The unfolding arranges the tensor so that mode-``mode`` fibres become
    rows. For a 3-D tensor ``X[m, t, d]``:

    * mode 0 → ``X[m, t, d] -> X[m, td]`` (each row is a vector of size t·d).
    * mode 1 → ``X[m, t, d] -> X[t, md]`` (each row is a vector of size m·d).
    * mode 2 → ``X[m, t, d] -> X[d, mt]`` (each row is a vector of size m·t).
    """
    if x.dim() != 3:
        raise ValueError(f"mode_n_unfold expects a 3-D tensor, got {tuple(x.shape)}")
    return torch.moveaxis(x, mode, 0).reshape(x.shape[mode], -1)


def mode_n_fold(mat: torch.Tensor, mode: int, shape: tuple[int, int, int]) -> torch.Tensor:
    """Inverse of :func:`mode_n_unfold`.

    ``mode_n_unfold(mode)`` lays out the tensor as ``(shape[mode], shape[(mode+1)%3], shape[(mode+2)%3])``
    when ``mode == 0`` (the simple case where the moved axis already sits at
    position 0), but for ``mode == 1`` and ``mode == 2`` the trailing two axes
    are reversed. We pick the correct order here.
    """
    if len(shape) != 3:
        raise ValueError(f"shape must have 3 dims, got {shape}")
    if mode == 0:
        full = mat.reshape(shape[0], shape[1], shape[2])
    elif mode == 1:
        full = mat.reshape(shape[1], shape[0], shape[2])
    else:
        full = mat.reshape(shape[2], shape[0], shape[1])
    return torch.moveaxis(full, 0, mode).contiguous()


def reconstruct_partial_tucker(
    factors: TuckerFactors,
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Reconstruct ``X̂`` from partial Tucker factors.

    Args:
        factors: output of :func:`partial_tucker_st_hosvd`.
        target_shape: original tensor shape ``(m, T, dh)``.

    Returns:
        Tensor of shape ``target_shape`` approximating the original input.
    """
    m, t, d = target_shape
    rt, rd = factors.r_token, factors.r_feature
    if factors.core.shape != (m, rt, rd):
        raise ValueError(
            f"core shape {factors.core.shape} != expected {(m, rt, rd)}"
        )
    if factors.u_token.shape != (t, rt):
        raise ValueError(
            f"u_token shape {factors.u_token.shape} != expected {(t, rt)}"
        )
    if factors.u_feature.shape != (d, rd):
        raise ValueError(
            f"u_feature shape {factors.u_feature.shape} != expected {(d, rd)}"
        )
    # X̂[m, t, d] = sum_{rt, rd} core[m, rt, rd] * U_T[t, rt] * U_d[d, rd]
    # Index labels: m/t/d are actual axes, a = rT (token rank), r = rd (feature
    # rank). Einsum contracts over the rank labels a and r.
    return torch.einsum(
        "mar,ta,dr->mtd",
        factors.core,
        factors.u_token,
        factors.u_feature,
    )


def partial_tucker_st_hosvd(
    x: torch.Tensor,
    *,
    r_token: int | None = None,
    r_feature: int | None = None,
    svd: SVD | None = None,
) -> TuckerFactors:
    """Sequentially truncated HOSVD for partial Tucker decomposition.

    The head/layer axis (mode 0) is *pinned*: it receives an identity factor
    and is never truncated.

    Args:
        x: tensor of shape ``(m, T, dh)``.
        r_token: target rank on the token mode. ``None`` keeps all.
        r_feature: target rank on the feature mode. ``None`` keeps all.
        svd: shared :class:`SVD` instance (so ``seed`` is shared across calls).

    Returns:
        :class:`TuckerFactors`.
    """
    if x.dim() != 3:
        raise ValueError(f"x must be 3-D, got {tuple(x.shape)}")
    m, t, d = x.shape

    rt_max = t
    rd_max = d
    rt = rt_max if r_token is None else max(1, min(int(r_token), rt_max))
    rd = rd_max if r_feature is None else max(1, min(int(r_feature), rd_max))

    svd = svd or SVD()
    rt_used = min(rt, t)
    rd_used = min(rd, d)

    # Mode 2 (feature) first — ST-HOSVD convention is to truncate the smaller
    # mode first to bound numerical cost. Either order works mathematically;
    # we follow the common "features first, then tokens" convention.
    feat_unfold = mode_n_unfold(x, 2)  # (d, m*T)
    feat_svd = svd(feat_unfold, rank=rd_used)
    u_dh = feat_svd.u  # (d, rd)
    s_dh = feat_svd.s
    feature_tail_mass = feat_svd.tail_mass

    # Project along mode 2.
    x_proj_dh = torch.einsum("mtd,dr->mtr", x, u_dh)

    # Mode 1 (token) next.
    tok_unfold = mode_n_unfold(x_proj_dh, 1)  # (T, m*rd)
    tok_svd = svd(tok_unfold, rank=rt_used)
    u_t = tok_svd.u  # (T, rt)
    s_t = tok_svd.s
    token_tail_mass = tok_svd.tail_mass

    # Project along mode 1.
    core = torch.einsum("mtr,tn->mnr", x_proj_dh, u_t)

    log.debug(
        "partial_tucker_st_hosvd: shape=%s ranks=(%d, %d) tail=(%.4e, %.4e)",
        tuple(x.shape),
        rt_used,
        rd_used,
        token_tail_mass,
        feature_tail_mass,
    )
    return TuckerFactors(
        core=core.contiguous(),
        u_token=u_t.contiguous(),
        u_feature=u_dh.contiguous(),
        token_sv=s_t,
        feature_sv=s_dh,
        token_tail_mass=token_tail_mass,
        feature_tail_mass=feature_tail_mass,
    )


def estimate_token_rank_for_budget(
    x: torch.Tensor,
    *,
    budget: int,
    bits_residual: int = 0,
    target_ratio: float = 1.0,
    feature_rank: int | None = None,
    dtype_bytes: int = 2,
) -> int:
    """Pick a token rank that fits ``budget`` bytes given a feature rank.

    Used by the allocator's offline grid search.

    Args:
        x: tensor of shape ``(m, T, dh)``.
        budget: target bytes.
        bits_residual: residual bit-width.
        target_ratio: unused (kept for future signed budgets).
        feature_rank: pinned feature rank.
        dtype_bytes: bytes per stored scalar (2 for fp16).

    Returns:
        Integer token rank ≤ ``T``.
    """
    m, t, d = x.shape
    rd = min(int(feature_rank) if feature_rank else d, d)
    # Cost: m*rT*rd + T*rT + d*rd  (in scalars; multiply by dtype_bytes) +
    # (bits/8) * m * T * d  for the residual.
    def cost(rt: int) -> int:
        scalars = m * rt * rd + t * rt + d * rd
        residual = (bits_residual // 8) * m * t * d
        return scalars * dtype_bytes + residual
    rt = t
    while rt > 1 and cost(rt) > budget:
        rt -= 1
    return rt


__all__ = [
    "TuckerFactors",
    "estimate_token_rank_for_budget",
    "mode_n_fold",
    "mode_n_unfold",
    "partial_tucker_st_hosvd",
    "reconstruct_partial_tucker",
]