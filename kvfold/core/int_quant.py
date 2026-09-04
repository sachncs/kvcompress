"""INT-only KV compressor — pure int quantisation baseline.

Used for ablation against JoLT: this is what an int2 / int4 / int8 quantizer
gives you without any Tucker back-bone. Matches the KIVI / TurboQuant
class in the paper's Table 2 baseline column.

The compressor reshapes K and V to 2-D ``(N, dh)`` and quantises along
the last axis. The bit-width is the *only* compression knob — there's no
rank or feature-budget trade-off. This is the right comparison for
"what if I just quantise and skip the Tucker back-bone?".
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from kvfold.config import Int8Config
from kvfold.core.base import Compressor, Payload, Stats
from kvfold.core.quant import IntQuant as IntQuantImpl
from kvfold.core.quant import dequantize_tensor, quantize_tensor

__all__ = ["IntQuant"]


log = logging.getLogger(__name__)


class IntQuant(Compressor):
    """Int-only baseline; per-cell per-channel quantisation.

    Note: this class name is the public compressor name. The ``int2``,
    ``int4``, ``int8`` method dispatch binds this class via the
    :class:`kvfold.core.builtins` registration with the appropriate
    ``bits`` kwarg.

    Attributes:
        bits: bit-width (2, 4, or 8).
        symmetric: symmetric vs. asymmetric integer quantisation.
        per_channel: per-channel vs per-tensor scales.
        group_size: optional per-group scale size.
    """

    method: str = "int_quant"

    def __init__(
        self,
        *,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = True,
        group_size: int | None = None,
    ) -> None:
        self.bits = int(bits)
        if self.bits not in (2, 4, 8):
            raise ValueError(f"IntQuant.bits must be 2/4/8; got {self.bits}")
        self.symmetric = bool(symmetric)
        self.per_channel = bool(per_channel)
        self.group_size = group_size
        self._quant = IntQuantImpl(
            bits=self.bits, symmetric=self.symmetric, per_channel=self.per_channel, group_size=self.group_size
        )

    @classmethod
    def default_config(cls) -> Int8Config:
        return Int8Config()

    def compress(self, key: torch.Tensor, value: torch.Tensor) -> tuple[Payload, Payload]:
        """Reshape to 2-D and quantise each side."""
        Compressor.validate(key, value)
        kp = self._side(key)
        vp = self._side(value)
        return kp, vp

    def restore(self, key_payload: Payload, value_payload: Payload) -> tuple[torch.Tensor, torch.Tensor]:
        k = dequantize_tensor(
            key_payload.data,
            dtype=f"int{self.bits}",
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            group_size=self.group_size,
            output_shape=key_payload.shape,
            output_dtype=key_payload.dtype,
        )
        v = dequantize_tensor(
            value_payload.data,
            dtype=f"int{self.bits}",
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            group_size=self.group_size,
            output_shape=value_payload.shape,
            output_dtype=value_payload.dtype,
        )
        return k, v

    def _side(self, x: torch.Tensor) -> Payload:
        flat = x.reshape(-1, x.shape[-1])
        q = self._quant.quantize(flat)
        return Payload(
            method=f"int{self.bits}",
            shape=tuple(x.shape),
            dtype=x.dtype,
            metadata={
                "bits": self.bits,
                "symmetric": self.symmetric,
                "per_channel": self.per_channel,
                "group_size": self.group_size,
            },
            data={
                "q": q[0],
                "scale": q[1],
                "zero_point": q[2],
                "original_last": torch.tensor(flat.shape[-1], dtype=torch.int32),
            },
            stats=Stats(
                bytes_original=x.numel() * x.element_size(),
                bytes_compressed=q[0].numel() * q[0].element_size() + q[1].numel() * q[1].element_size() + q[2].numel() * q[2].element_size() + 4,
            ),
        )
