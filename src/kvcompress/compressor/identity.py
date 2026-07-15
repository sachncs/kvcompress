"""Identity compressor — passthrough with no actual compression.

Useful as a baseline for ablation studies: it stores K and V in fp16
(half the size of fp32, same as fp16 model weights) so memory accounting
is realistic but no JoLT-specific operations are performed.
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

log = logging.getLogger(__name__)


class IdentityCompressor(KVCompressor):
    """No-op compressor that stores K/V in fp16."""

    name = "identity"

    def __init__(self, *, factor_dtype: torch.dtype = torch.float16, **_unused: Any) -> None:
        super().__init__()
        self.factor_dtype = factor_dtype

    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[CompressedPayload, CompressedPayload]:
        kp = CompressedPayload(
            method="identity",
            shape=tuple(key.shape),
            dtype=key.dtype,
            metadata={"r_token": 0, "r_feature": 0, "bits": 0},
            data={"value": key.to(self.factor_dtype).contiguous()},
            stats=CompressorStats(
                bytes_original=key.numel() * key.element_size(),
                bytes_compressed=key.numel() * self.factor_dtype.itemsize,
            ),
        )
        vp = CompressedPayload(
            method="identity",
            shape=tuple(value.shape),
            dtype=value.dtype,
            metadata={"r_token": 0, "r_feature": 0, "bits": 0},
            data={"value": value.to(self.factor_dtype).contiguous()},
            stats=CompressorStats(
                bytes_original=value.numel() * value.element_size(),
                bytes_compressed=value.numel() * self.factor_dtype.itemsize,
            ),
        )
        return kp, vp

    def decompress(
        self,
        key_payload: CompressedPayload,
        value_payload: CompressedPayload,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = key_payload.data["value"].to(key_payload.dtype)
        v = value_payload.data["value"].to(value_payload.dtype)
        return k, v


__all__ = ["IdentityCompressor"]