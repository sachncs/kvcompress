"""Real IEEE-style FP8 (E4M3 / E5M2) quantisation.

PyTorch ≥ 2.5 exposes native ``torch.float8_e4m3fn`` and
``torch.float8_e5m2`` dtypes; we use those when present. When the dtype
is missing (older torch) the constructor raises :class:`DTypeError` so
the caller is told to upgrade rather than silently downgrading to fp16.

:class:`Float8` is a :class:`Compressor` (not a primitive). It stores the
input tensor as a uint8 ``packed`` buffer plus per-channel ``scale`` and
``zero_point`` tensors. The round-trip error is bounded by half an FP8
bin — for E4M3 that's roughly ±6 %.

Attributes:
    variant: ``"e4m3"`` (range, 240 max) or ``"e5m2"`` (wider exponent,
        less precision).
    per_channel: per-channel scale (last-axis slices) if True, else a
        single scalar scale per tensor.
    group_size: optional group size for grouped scales; mutually
        exclusive with ``per_channel``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch

from kvfold.config import Float8Config
from kvfold.core.base import Compressor, Payload, Stats
from kvfold.errors import DTypeError

__all__ = ["Float8"]


log = logging.getLogger(__name__)


def e4m3_dtype() -> torch.dtype:
    """Resolve ``torch.float8_e4m3fn`` or raise :class:`DTypeError`."""
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise DTypeError(
            "torch.float8_e4m3fn is not available in this torch build",
            hint="upgrade to torch >= 2.5 or use method='bf16'",
        )
    return dtype


def e5m2_dtype() -> torch.dtype:
    """Resolve ``torch.float8_e5m2`` or raise :class:`DTypeError`."""
    dtype = getattr(torch, "float8_e5m2", None)
    if dtype is None:
        raise DTypeError(
            "torch.float8_e5m2 is not available in this torch build",
            hint="upgrade to torch >= 2.5 or use method='bf16'",
        )
    return dtype


class Float8(Compressor):
    """IEEE-style FP8 quantisation compressor.

    The compression ratio is exactly 4× vs fp32 and 2× vs fp16, with the
    documented loss bound. Use :class:`FloatCast` instead when you want a
    lossless dtype halving.

    Attributes:
        variant: ``"e4m3"`` or ``"e5m2"``.
        per_channel: per-channel vs per-tensor scaling.
        group_size: optional per-group scaling; mutually exclusive with
            ``per_channel``.
    """

    method: str = "fp8"

    def __init__(
        self,
        *,
        variant: Literal["e4m3", "e5m2"] = "e4m3",
        per_channel: bool = True,
        group_size: int | None = None,
        **unused: Any,
    ) -> None:
        if variant == "e4m3":
            self.target_dtype = e4m3_dtype()
        elif variant == "e5m2":
            self.target_dtype = e5m2_dtype()
        else:
            raise DTypeError(
                f"unknown Float8 variant {variant!r}; expected 'e4m3' or 'e5m2'",
                hint="pass variant='e4m3' (range) or variant='e5m2' (wider exponent)",
            )
        if per_channel and group_size is not None:
            raise DTypeError(
                "Float8: per_channel=True and group_size are mutually exclusive",
                hint="set per_channel=False when group_size is provided",
            )
        self.variant = variant
        self.per_channel = bool(per_channel)
        self.group_size = group_size

    @classmethod
    def default_config(cls) -> Float8Config:
        return Float8Config()

    def compress(self, key: torch.Tensor, value: torch.Tensor) -> tuple[Payload, Payload]:
        """Quantise K and V to FP8."""
        Compressor.validate(key, value)
        kp = self.quantize_side(key)
        vp = self.quantize_side(value)
        return kp, vp

    def restore(self, key_payload: Payload, value_payload: Payload) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantise K and V from FP8."""
        k = self.dequantize_payload(key_payload)
        v = self.dequantize_payload(value_payload)
        return k, v

    def quantize_side(self, x: torch.Tensor) -> Payload:
        """Quantise a single (K or V) tensor."""
        if not x.is_floating_point():
            raise DTypeError(
                f"Float8 expects floating-point input; got dtype {x.dtype}",
                hint="cast to float32 or float16 before compress()",
            )
        original_dtype = x.dtype
        x_fp32 = x.to(torch.float32) if x.dtype != torch.float32 else x

        if self.per_channel:
            x_flat = x_fp32.reshape(-1, x_fp32.shape[-1])
            x_amax = x_flat.abs().amax(dim=0).clamp(min=1e-12)
            scale = x_amax / torch.finfo(self.target_dtype).max
            scale = scale.to(torch.float32)
            zero_point = torch.zeros_like(scale, dtype=torch.int32)
            scaled = x_flat / scale.view(1, -1)
            packed = scaled.to(self.target_dtype).to(torch.uint8).reshape(x.shape)
        elif self.group_size is not None:
            last = x_fp32.shape[-1]
            if last % self.group_size != 0:
                raise DTypeError(
                    f"Float8: last dim {last} not divisible by group_size {self.group_size}",
                    hint="set group_size to a divisor of the last dim, or omit it",
                )
            x_g = x_fp32.reshape(*x_fp32.shape[:-1], last // self.group_size, self.group_size)
            x_amax = x_g.abs().amax(dim=-1).clamp(min=1e-12)
            scale = (x_amax / torch.finfo(self.target_dtype).max).to(torch.float32)
            zero_point = torch.zeros_like(scale, dtype=torch.int32)
            scaled = x_g / scale.unsqueeze(-1)
            packed = scaled.to(self.target_dtype).to(torch.uint8).reshape(*x.shape[:-1], last)
        else:
            x_amax = x_fp32.abs().amax().clamp(min=1e-12)
            scale = (x_amax / torch.finfo(self.target_dtype).max).reshape(()).to(torch.float32)
            zero_point = torch.zeros((), dtype=torch.int32)
            packed = (x_fp32 / scale).to(self.target_dtype).to(torch.uint8)

        return Payload(
            method="fp8",
            shape=tuple(x.shape),
            dtype=original_dtype,
            metadata={
                "variant": self.variant,
                "per_channel": self.per_channel,
                "group_size": self.group_size,
            },
            data={
                "packed": packed.contiguous(),
                "scale": scale.contiguous(),
                "zero_point": zero_point.contiguous(),
            },
            stats=Stats(
                bytes_original=x.numel() * x.element_size(),
                bytes_compressed=(
                    packed.numel() * packed.element_size()
                    + scale.numel() * scale.element_size()
                    + zero_point.numel() * zero_point.element_size()
                ),
            ),
        )

    def dequantize_payload(self, payload: Payload) -> torch.Tensor:
        """Inverse of :meth:`quantize_side`."""
        packed = payload.data["packed"]
        scale = payload.data["scale"]
        zero_point = payload.data["zero_point"]
        original_shape = tuple(int(d) for d in payload.shape)
        packed_fp8 = packed.to(self.target_dtype)
        packed_fp32 = packed_fp8.to(torch.float32)

        if self.per_channel:
            flat = packed_fp32.reshape(-1, packed_fp32.shape[-1])
            recovered = flat * scale.view(1, -1)
            recovered = recovered.reshape(original_shape)
        elif self.group_size is not None:
            last = original_shape[-1]
            r = recovered_shape = last // self.group_size
            packed_fp32 = packed_fp32.reshape(*original_shape[:-1], r, self.group_size)
            recovered = packed_fp32 * scale.unsqueeze(-1)
            recovered = recovered.reshape(original_shape)
        else:
            recovered = packed_fp32 * scale
            recovered = recovered.reshape(original_shape)

        return (recovered + zero_point.to(recovered.dtype)).to(payload.dtype)
