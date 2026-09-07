"""Flash compressor — randomized-SVD JoLT variant.

Same algorithm as :class:`Jolt` but replaces the exact token-mode SVD
with a randomized low-rank SVD capped at ``q_cap``. The cap is chosen by
the configured :class:`CapPolicy`.

Default policy (paper-faithful): ``q_cap = min(max(q_min(R), ⌈T / 32⌉), 512``
with ``q_min = 32`` for ``R ≤ 4`` and ``q_min = 64`` for ``R ≥ 5``. The
policy is a no-op for ``T ≤ 1024`` so every short-context number is
unchanged, and grows the cap sublinearly beyond that.

Speed: per the paper, Flash delivers 5-13× compression-time speedup
over exact JoLT at matched quality (paper Section 5).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Literal

import torch

from kvfold.config import FlashConfig
from kvfold.core.jolt import Jolt, JoLTFactors as JoltFactors
from kvfold.core.residual import encode_residual
from kvfold.core.svd import Exact, Randomized
from kvfold.core.tucker import partial_tucker_st_hosvd, reconstruct_partial_tucker

__all__ = ["CapPolicy", "Flash", "LinearCap"]


log = logging.getLogger(__name__)


class CapPolicy(ABC):
    """Strategy for choosing the token-mode SVD cap.

    The cap is the upper bound on the randomised-SVD sketch size; the
    policy converts ``(context_length, ratio)`` into an integer in
    ``[1, 512]``. Concrete strategies: :class:`LinearCap` (paper-faithful).
    """

    name: Literal["linear"] = "linear"

    @abstractmethod
    def cap(self, context_length: int, ratio: float) -> int: ...


class LinearCap(CapPolicy):
    """Paper-faithful cap: ``min(max(q_min(R), ⌈T/32⌉), 512)``.

    ``q_min = 32`` for ``R ≤ 4`` and ``q_min = 64`` for ``R ≥ 5``. The
    policy is a no-op for ``T ≤ 1024`` (where ``q_min`` is the floor).
    """

    name: Literal["linear"] = "linear"

    def cap(self, context_length: int, ratio: float) -> int:
        q_min = 64 if ratio >= 5.0 else 32
        cap = max(q_min, math.ceil(context_length / 32))
        return min(cap, 512)


class Flash(Jolt):
    """Randomized-SVD JoLT variant.

    Inherits the allocator, residual path, and stats from :class:`Jolt`.
    Replaces the token-mode SVD with a :class:`Randomized` decomposer
    subject to the configured :class:`CapPolicy`.

    Attributes:
        cap: explicit cap override (``None`` means auto).
        cap_policy: strategy used when ``cap`` is ``None``.
    """

    method: str = "flash"

    def __init__(
        self,
        *,
        ratio: float = 3.0,
        bits: tuple[int, ...] = (0, 2, 4, 8),
        cap: int | None = None,
        cap_policy: Literal["linear"] = "linear",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            ratio=ratio,
            bits=bits,
            seed=seed,
            **kwargs,
        )
        self.cap = int(cap) if cap is not None else None
        self.cap_policy: CapPolicy = LinearCap()

    @classmethod
    def default_config(cls) -> FlashConfig:
        return FlashConfig()

    def compress_cell(self, x: torch.Tensor, allocation: Any) -> JoLTFactors:
        """Run partial Tucker + JL residual with a randomised token-mode SVD.

        Overrides :meth:`Jolt.compress_cell` to use a per-call randomised
        decomposer whose cap is determined by ``(x.shape[1], self.ratio)``
        via :attr:`cap_policy`.
        """
        if x.dim() != 3:
            raise ValueError(f"Flash expects 3-D (m, T, dh); got {tuple(x.shape)}")
        if self.cap is None:
            cap = self.cap_policy.cap(x.shape[1], self.ratio)
        else:
            cap = self.cap
        token_decomposer = Randomized(seed=self.seed, cap=cap)
        tucker = partial_tucker_st_hosvd(
            x,
            r_token=allocation.r_token,
            r_feature=allocation.r_feature,
            svd=token_decomposer,
        )
        recon = reconstruct_partial_tucker(tucker, x.shape)
        residual_tensor = (x - recon).contiguous() if allocation.bits > 0 else None

        residual: Any = None
        if allocation.bits > 0 and residual_tensor is not None:
            residual = encode_residual(
                residual_tensor,
                bits=allocation.bits,
                seed=self.seed,
                distribution=self.jl_distribution,
                symmetric=self.symmetric_quant,
                per_channel=self.per_channel_quant,
                group_size=self.group_size,
            )
        return JoltFactors(
            tucker=tucker,
            residual=residual,
            allocation=allocation,
        )
