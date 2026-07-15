"""INT-only KV compressor — pure quantization baseline.

Used for ablation against JoLT: this is what an int4 / int8 quantizer
gives you without any Tucker back-bone. Matches the "KIVI" / "TurboQuant"
class in the paper's Table 2 baseline column.

The compressor reshapes K and V to 2-D ``(N, dh)`` and quantises along
the last axis. The bit-width is the *only* compression knob — there's no
rank or feature-budget trade-off. This is the right comparison for "what
if I just quantise and skip the Tucker back-bone?".
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
from kvcompress.compressor.quantization import (
    IntQuantizer,
    dequantize_tensor,
    quantize_tensor,
)

log = logging.getLogger(__name__)


class IntQuantOnlyCompressor(KVCompressor):
    """Per-channel integer quantization, no Tucker.

    Args:
        bits: ``2``, ``4``, or ``8``.
        per_channel: per-channel scales if True.
        symmetric: symmetric vs. asymmetric.
    """

    name = "int-quant-only"

    def __init__(
        self,
        *,
        bits: int = 4,
        per_channel: bool = True,
        symmetric: bool = True,
        **_unused: Any,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.quantizer = IntQuantizer(bits=bits, symmetric=symmetric, per_channel=per_channel)

    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[CompressedPayload, CompressedPayload]:
        return self._compress_one(key), self._compress_one(value)

    def decompress(
        self,
        key_payload: CompressedPayload,
        value_payload: CompressedPayload,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = self._decompress_one(key_payload)
        v = self._decompress_one(value_payload)
        return k, v

    def _compress_one(self, x: torch.Tensor) -> CompressedPayload:
        # Reshape 3-D to 2-D for per-channel quantization along the last axis.
        original_shape = tuple(x.shape)
        flat = x.reshape(-1, original_shape[-1])
        payload = quantize_tensor(
            flat,
            dtype=f"int{self.bits}",
            symmetric=self.symmetric,
            per_channel=self.per_channel,
        )
        return CompressedPayload(
            method=f"int{self.bits}-only",
            shape=original_shape,
            dtype=x.dtype,
            metadata={
                "bits": self.bits,
                "symmetric": self.symmetric,
                "per_channel": self.per_channel,
                "original_last": int(payload["original_last"].item()),
            },
            data={
                "q": payload["q"],
                "scale": payload["scale"],
                "zero_point": payload["zero_point"],
                "original_last": payload["original_last"],
            },
            stats=CompressorStats(
                bytes_original=x.numel() * x.element_size(),
                bytes_compressed=payload["q"].numel() * payload["q"].element_size()
                + payload["scale"].numel() * payload["scale"].element_size()
                + payload["zero_point"].numel() * payload["zero_point"].element_size(),
            ),
        )

    def _decompress_one(self, payload: CompressedPayload) -> torch.Tensor:
        flat = dequantize_tensor(
            {
                "q": payload.data["q"],
                "scale": payload.data["scale"],
                "zero_point": payload.data["zero_point"],
                "original_last": payload.data["original_last"],
            },
            dtype=f"int{payload.metadata['bits']}",
            symmetric=payload.metadata["symmetric"],
            per_channel=payload.metadata["per_channel"],
            output_dtype=payload.dtype,
        )
        return flat.reshape(payload.shape)


__all__ = ["IntQuantOnlyCompressor"]
