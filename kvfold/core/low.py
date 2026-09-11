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

Algorithm (per side)
--------------------

For each ``X ∈ R^{m·T × dh}`` we form the unfold ``X̄ = X.reshape(m·T, dh)``
and compute a truncated SVD::

    X̄ ≈ U · diag(S) · Vh      with U ∈ R^{m·T×r}, Vh ∈ R^{r×dh}

We store ``U`` and ``Vh`` plus the singular values ``S``. Storage::

    B = (m·T·r + r + r·dh) · c     (c = sizeof(dtype) in bytes)

vs. raw ``B_raw = m·T·dh · sizeof(dtype)`` — so the achieved ratio is
roughly ``(m·T·dh) / (m·T·r + r·dh)`` for ``c=1``, dropping further when
``c`` is fp16. Reconstruction is::

    X̂ = (U · diag(S)) @ Vh      reshaped back to (m, T, dh)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from kvfold.config import LowRankConfig
from kvfold.core.base import Compressor, Payload, Stats
from kvfold.core.svd import Exact

__all__ = ["Low"]


log = logging.getLogger(__name__)


class Low(Compressor):
    """Low-rank matrix SVD compressor.

    Treats each K/V tensor as a 2-D matrix ``(m·T, dh)`` and applies a
    truncated SVD at the requested rank.

    Attributes:
        rank: target rank.
        dtype: stored factor dtype.
    """

    method: str = "low"

    def __init__(
        self,
        *,
        rank: int = 64,
        dtype: torch.dtype = torch.float16,
        seed: int = 0,
        **unused: Any,
    ) -> None:
        if unused:
            from kvfold.errors import MethodConfigError
            raise MethodConfigError(
                "low",
                ",".join(sorted(unused.keys())),
                "unknown field(s); see LowRankConfig for valid kwargs",
            )
        self.rank = int(rank)
        self.dtype = dtype
        self.decomposer = Exact()

    @classmethod
    def default_config(cls) -> LowRankConfig:
        return LowRankConfig()

    def compress(self, key: torch.Tensor, value: torch.Tensor) -> tuple[Payload, Payload]:
        """Truncated-SVD compress K and V at the given rank."""
        Compressor.validate(key, value)
        m, T, dh = key.shape
        K_flat = key.reshape(m * T, dh)
        V_flat = value.reshape(m * T, dh)
        k_res = self.decomposer.decompose(K_flat, rank=self.rank)
        v_res = self.decomposer.decompose(V_flat, rank=self.rank)
        kp = self.build_payload(key, k_res, m, T, dh)
        vp = self.build_payload(value, v_res, m, T, dh)
        return kp, vp

    def restore(self, key_payload: Payload, value_payload: Payload) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of :meth:`compress`."""
        k = self.reconstruct_payload(key_payload)
        v = self.reconstruct_payload(value_payload)
        return k, v

    def build_payload(
        self,
        original: torch.Tensor,
        decomposition: Any,
        m: int,
        T: int,
        dh: int,
    ) -> Payload:
        """Wrap a single-side decomposition result into a serialisable payload."""
        u = decomposition.u.to(self.dtype).contiguous()
        s = decomposition.s.to(self.dtype).contiguous()
        vh = decomposition.vh.to(self.dtype).contiguous()
        return Payload(
            method="low",
            shape=tuple(original.shape),
            dtype=original.dtype,
            metadata={"rank": decomposition.rank, "m": m, "T": T, "dh": dh},
            data={"u": u, "s": s, "vh": vh},
            stats=Stats(
                bytes_original=original.numel() * original.element_size(),
                bytes_compressed=u.numel() * u.element_size()
                + s.numel() * s.element_size()
                + vh.numel() * vh.element_size(),
            ),
        )

    def reconstruct_payload(self, payload: Payload) -> torch.Tensor:
        """Reconstruct a single side from a :class:`Payload`."""
        u = payload.data["u"].to(torch.float32)
        s = payload.data["s"].to(torch.float32)
        vh = payload.data["vh"].to(torch.float32)
        flat = (u * s) @ vh
        m = int(payload.metadata["m"])
        T = int(payload.metadata["T"])
        return flat.reshape(m, T, -1).to(payload.dtype)
