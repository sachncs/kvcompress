"""Rank selection strategies for partial Tucker decomposition.

A :class:`RankStrategy` decides ``r_token`` (and optionally
``r_feature``) for a given input tensor and budget. Concrete strategies:

* :class:`TailMass` — pick the smallest rank whose discarded tail mass
  is below ``epsilon``.
* :class:`Fixed` — use the caller-supplied rank.
* :class:`Adaptive` — pick rank to meet a byte ``budget``.

Used by the budget allocator to propose per-cell ranks; the allocator
still chooses among the proposed candidates.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch

from kvfold.core.svd import Exact

log = logging.getLogger(__name__)


class RankStrategy(ABC):
    """Abstract strategy for choosing per-cell ``r_token`` (and optionally ``r_feature``)."""

    @abstractmethod
    def rank(self, x: torch.Tensor, *, epsilon: float = 0.1, feature_rank: int | None = None) -> tuple[int, int]:
        """Return ``(r_token, r_feature)`` for tensor ``x``."""


class Fixed(RankStrategy):
    """Fixed ``r_token`` regardless of spectrum."""

    def __init__(self, r_token: int = 32, r_feature: int | None = None) -> None:
        self.r_token = int(r_token)
        self.r_feature = r_feature

    def rank(self, x: torch.Tensor, *, epsilon: float = 0.1, feature_rank: int | None = None) -> tuple[int, int]:
        rd = self.r_feature if self.r_feature is not None else (feature_rank if feature_rank is not None else min(x.shape[-1], 64))
        return self.r_token, int(rd)


class TailMass(RankStrategy):
    """Smallest rank whose discarded tail mass is below ``epsilon``.

    The tail mass is computed from the token-mode unfolding of ``x`` via
    :class:`Exact`'s full SVD.
    """

    def __init__(self, decomposer: Exact | None = None) -> None:
        self.decomposer = decomposer or Exact()

    def rank(self, x: torch.Tensor, *, epsilon: float = 0.1, feature_rank: int | None = None) -> tuple[int, int]:
        if x.dim() != 3:
            raise ValueError(f"TailMass expects 3-D (m, T, dh); got {tuple(x.shape)}")
        m, T, dh = x.shape
        token_unfold = x.transpose(0, 1).reshape(T, m * dh)
        decomp = self.decomposer.decompose(token_unfold, rank=min(T, m * dh))
        s = decomp.s
        total = float((s * s).sum())
        if total <= 0:
            return 1, min(dh, 8)
        cumulative = float((s * s).cumsum(dim=0)[-1].item())
        retained_frac = 1.0 - epsilon
        rt = 1
        for k in range(s.shape[0]):
            cumsum = float((s[: k + 1] * s[: k + 1]).sum().item())
            if cumsum / total >= retained_frac:
                rt = k + 1
                break
        rt = max(1, min(rt, T))
        rd = min(dh, max(8, rt // 2))
        return rt, rd


class Adaptive(RankStrategy):
    """Pick rank to fit a byte budget for the Tuck + residual storage."""

    def __init__(self, target_ratio: float = 3.0, factor_dtype_bytes: int = 2) -> None:
        if target_ratio <= 1.0:
            raise ValueError(f"Adaptive target_ratio must be > 1.0; got {target_ratio}")
        self.target_ratio = float(target_ratio)
        self.factor_dtype_bytes = int(factor_dtype_bytes)

    def rank(self, x: torch.Tensor, *, epsilon: float = 0.1, feature_rank: int | None = None) -> tuple[int, int]:
        if x.dim() != 3:
            raise ValueError(f"Adaptive expects 3-D (m, T, dh); got {tuple(x.shape)}")
        m, T, dh = x.shape
        original_bytes = m * T * dh * 4
        budget = original_bytes / self.target_ratio
        rd = feature_rank if feature_rank is not None else min(dh, 64)
        tucker_bytes = (m * rd + T * rd + m * T * rd) * self.factor_dtype_bytes
        rt = max(1, min(T, int((budget - tucker_bytes) / max(m * rd, 1) / self.factor_dtype_bytes)))
        return rt, rd


__all__ = ["Adaptive", "Fixed", "RankStrategy", "TailMass"]
