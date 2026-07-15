"""Low-rank matrix SVD compressor.

Baseline compressor that stores K and V as a matrix SVD: ``K ≈ U_K S_K V_Kᵀ``,
``V ≈ U_V S_V V_Vᵀ`` with rank truncation. Used for ablation against JoLT.

Why this baseline exists:

* JoLT compresses along two axes (token and feature). The natural
  comparison is a method that compresses along one axis — the matrix SVD
  here compresses the (m·T, dh) matrix, which keeps all ``T`` tokens but
  reduces the per-head dim. This isolates the contribution of the
  two-axis Tucker vs. the residual path.
* Same byte accounting as JoLT, so cross-method comparisons are direct.

This is the natural ``RankCompression`` baseline against which
``TuckerCompression`` was shown to win by ~5-10× in the paper's
Table 2.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from kvcompress.compressor.base import (
    CompressedPayload,
    CompressorStats,
    KVCompressor,
)
from kvcompress.compressor.svd import SVD

log = logging.getLogger(__name__)


class LowRankCompressor(KVCompressor):
    """Low-rank matrix SVD compressor.

    Treats each K/V tensor as a 2-D matrix ``(m·T, dh)`` and applies a
    truncated SVD at the requested rank.

    Args:
        rank: target rank.
        factor_dtype: stored factor dtype (``fp16`` or ``fp32``).
    """

    name = "lowrank"

    def __init__(
        self,
        *,
        rank: int = 64,
        factor_dtype: torch.dtype = torch.float16,
        seed: int = 0,
        **_unused: Any,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.factor_dtype = factor_dtype
        self.svd = SVD(method="exact", seed=seed)

    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[CompressedPayload, CompressedPayload]:
        if key.shape != value.shape:
            raise ValueError(f"K/V shape mismatch: {key.shape} vs {value.shape}")
        m, T, dh = key.shape
        K_flat = key.reshape(m * T, dh)
        V_flat = value.reshape(m * T, dh)
        k_res = self.svd(K_flat, rank=self.rank)
        v_res = self.svd(V_flat, rank=self.rank)
        kp = self._build_payload(key, k_res, m, T, dh)
        vp = self._build_payload(value, v_res, m, T, dh)
        return kp, vp

    def decompress(
        self,
        key_payload: CompressedPayload,
        value_payload: CompressedPayload,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = self._reconstruct_payload(key_payload)
        v = self._reconstruct_payload(value_payload)
        return k, v

    def _build_payload(
        self,
        original: torch.Tensor,
        svd_res: Any,
        m: int,
        T: int,
        dh: int,
    ) -> CompressedPayload:
        u = svd_res.u.to(self.factor_dtype).contiguous()
        s = svd_res.s.to(self.factor_dtype).contiguous()
        vh = svd_res.vh.to(self.factor_dtype).contiguous()
        return CompressedPayload(
            method="lowrank",
            shape=tuple(original.shape),
            dtype=original.dtype,
            metadata={"rank": svd_res.rank, "m": m, "T": T, "dh": dh},
            data={"u": u, "s": s, "vh": vh},
            stats=CompressorStats(
                bytes_original=original.numel() * original.element_size(),
                bytes_compressed=u.numel() * u.element_size()
                + s.numel() * s.element_size()
                + vh.numel() * vh.element_size(),
            ),
        )

    def _reconstruct_payload(self, payload: CompressedPayload) -> torch.Tensor:
        u = payload.data["u"].to(torch.float32)
        s = payload.data["s"].to(torch.float32)
        vh = payload.data["vh"].to(torch.float32)
        flat = (u * s) @ vh
        m = int(payload.metadata["m"])
        T = int(payload.metadata["T"])
        return flat.reshape(m, T, -1).to(payload.dtype)  # type: ignore[no-any-return]


__all__ = ["LowRankCompressor"]
