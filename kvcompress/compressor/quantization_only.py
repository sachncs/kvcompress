"""INT-only KV compressor — pure quantization baseline.

Used for ablation against JoLT: this is what an int4 / int8 quantizer
gives you without any Tucker back-bone. Matches the "KIVI" / "TurboQuant"
class in the paper's Table 2 baseline column.

The compressor reshapes K and V to 2-D ``(N, dh)`` and quantises along
the last axis. The bit-width is the *only* compression knob — there's no
rank or feature-budget trade-off. This is the right comparison for "what
if I just quantise and skip the Tucker back-bone?".

Algorithm (per side)
--------------------

For an input ``X ∈ R^{m × T × dh}``::

    X̄ = X.reshape(-1, dh)
    q, scale, zp, original_last = quantize_tensor(X̄, bits, ...)

Storage per element is ``bits/8`` bytes for the codeword plus the scale
and zero-point. For per-channel quantisation the scale and zero-point
are per-row (length ``dh`` instead of one scalar): this trades a tiny
amount of metadata for much lower reconstruction error.

The quantise/dequantise round-trip error is bounded by the resolution
of the integer grid: ``|X - dequantise(quantise(X))|∞ ≤ step / 2``
where ``step = (max - min) / (2^bits - 1)`` (asymmetric) or
``step = 2·|max| / (2^bits - 1)`` (symmetric). Compared to JoLT this
is much worse on the token axis because every token shares the same
``dh``-dim grid.
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

__all__ = ["IntQuantOnlyCompressor"]

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
        **unused: Any,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        self.per_channel = per_channel
        self.symmetric = symmetric
        # Eagerly construct the quantizer so failures surface at __init__
        # time rather than at the first compress() call.
        self.quantizer = IntQuantizer(bits=bits, symmetric=symmetric, per_channel=per_channel)

    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[CompressedPayload, CompressedPayload]:
        """Per-side int quantisation of K and V.

        Reshapes each side to ``(-1, dh)`` and applies the configured
        integer quantiser along the last axis. No Tucker decomposition,
        no residual, no JL rotation.

        Args:
            key: input K tensor of shape ``(m, T, dh)``.
            value: input V tensor of the same shape.

        Returns:
            Two :class:`CompressedPayload` carrying the per-channel
            quantised codewords plus the scale / zero-point tables.
        """
        return self.compress_one(key), self.compress_one(value)

    def decompress(
        self,
        key_payload: CompressedPayload,
        value_payload: CompressedPayload,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of :meth:`compress`.

        Dequantises both payloads back to ``payload.dtype`` and
        reshapes to ``payload.shape``.

        Args:
            key_payload: payload produced by :meth:`compress`.
            value_payload: payload produced by :meth:`compress`.

        Returns:
            ``(K, V)`` -- two tensors of the original dtype and shape.
        """
        k = self.decompress_one(key_payload)
        v = self.decompress_one(value_payload)
        return k, v

    def compress_one(self, x: torch.Tensor) -> CompressedPayload:
        """Quantise a single K/V tensor to a :class:`CompressedPayload`.

        Steps:
            1. Reshape ``x`` to ``(-1, dh)`` so the last axis is the
               feature axis (per-channel quantisation).
            2. Apply ``quantize_tensor`` to produce the integer codeword
               plus per-channel scales and zero-points.
            3. Wrap codeword + scales + zero-point into a payload and
               record the byte counts.

        Args:
            x: input tensor of any rank; the last axis is the quantised
               feature axis.

        Returns:
            A :class:`CompressedPayload` with ``data`` containing
            ``q``, ``scale``, ``zero_point``, and ``original_last``.
        """
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

    def decompress_one(self, payload: CompressedPayload) -> torch.Tensor:
        """Dequantise a single K/V payload back to a tensor.

        Args:
            payload: payload produced by :meth:`compress_one`.

        Returns:
            Tensor of ``payload.shape`` and ``payload.dtype``.
        """
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
