"""Concrete :class:`kvfold.core.base.Compressor` subclasses.

Each class registers itself with :data:`kvfold.core.dispatch.REGISTRY`
on import. The dispatcher routes the public ``method`` string to one of
these classes.

Classes:

* :class:`Jolt` — paper-faithful JoLT.
* :class:`Flash` — randomised-SVD JoLT with cap policy.
* :class:`Low` — pure low-rank matrix SVD baseline.
* :class:`IntQuant` — pure int quantisation (int2/int4/int8).
* :class:`Float8` — IEEE FP8 (E4M3/E5M2) quantisation.
* :class:`FloatCast` — dtype-only halving for fp16/bf16.
* :class:`Pass` — true passthrough (no compression).
"""

from __future__ import annotations

from typing import Any

import torch

from kvfold.config import (
    Bf16Config,
    FlashConfig,
    Float8Config,
    FloatConfig,
    Fp16Config,
    Int2Config,
    Int4Config,
    Int8Config,
    JoltConfig,
    LowRankConfig,
    PassConfig,
)
from kvfold.core.base import Compressor, Payload, Stats
from kvfold.core.dispatch import REGISTRY as DISPATCH_REGISTRY, register


__all__ = [
    "FloatCast",
    "Float8Compressor",
    "Jolt",
    "Flash",
    "Low",
    "IntQuant",
    "Pass",
]


class IntQuant(Compressor):
    """Pure int quantisation compressor (int2 / int4 / int8).

    Stores the input as a bit-packed uint8 buffer plus per-channel scale
    and zero-point. Reconstruction error is bounded by half a
    quantisation bin.
    """

    method: str = "int_quant"

    def __init__(self, *, bits: int = 8, symmetric: bool = True, per_channel: bool = True, group_size: int | None = None) -> None:
        from kvfold.core.quant import IntQuant as QuantImpl

        self.bits = int(bits)
        if self.bits not in (2, 4, 8):
            raise ValueError(f"IntQuant.bits must be 2/4/8; got {self.bits}")
        self.symmetric = bool(symmetric)
        self.per_channel = bool(per_channel)
        self.group_size = group_size
        self._quant = QuantImpl(bits=self.bits, symmetric=self.symmetric, per_channel=self.per_channel, group_size=self.group_size)

    @classmethod
    def default_config(cls) -> Int8Config:
        return Int8Config()

    def compress(self, key: torch.Tensor, value: torch.Tensor) -> tuple[Payload, Payload]:
        Compressor.validate(key, value)
        return self._side(key), self._side(value)

    def restore(self, key_payload: Payload, value_payload: Payload) -> tuple[torch.Tensor, torch.Tensor]:
        from kvfold.core.quant import dequantize_tensor
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
        from kvfold.core.quant import quantize_tensor
        data = quantize_tensor(
            x,
            dtype=f"int{self.bits}",
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            group_size=self.group_size,
        )
        return Payload(
            method=f"int{self.bits}",
            shape=tuple(x.shape),
            dtype=x.dtype,
            metadata={"bits": self.bits, "symmetric": self.symmetric, "per_channel": self.per_channel, "group_size": self.group_size},
            data=data,
            stats=Stats(bytes_original=x.numel() * x.element_size(), bytes_compressed=sum(v.numel() * v.element_size() for v in data.values() if isinstance(v, torch.Tensor))),
        )


class FloatCast(Compressor):
    """Dtype-only halving compressor (fp16 / bf16).

    Casts the input to a 16-bit float and stores it verbatim. The byte
    count halves; the numerical content is unchanged. Use this when you
    want to halve cache size without any algorithmic compression.
    """

    method: str = "fp_cast"

    def __init__(self, *, dtype: torch.dtype = torch.float16) -> None:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"FloatCast.dtype must be torch.float16 or torch.bfloat16; got {dtype}")
        self.dtype = dtype

    @classmethod
    def default_config(cls) -> Fp16Config:
        return Fp16Config()

    def compress(self, key: torch.Tensor, value: torch.Tensor) -> tuple[Payload, Payload]:
        Compressor.validate(key, value)
        kp = Payload(
            method="fp_cast",
            shape=tuple(key.shape),
            dtype=key.dtype,
            metadata={"dtype": str(self.dtype)},
            data={"value": key.to(self.dtype).contiguous()},
            stats=Stats(
                bytes_original=key.numel() * key.element_size(),
                bytes_compressed=key.numel() * self.dtype.itemsize,
            ),
        )
        vp = Payload(
            method="fp_cast",
            shape=tuple(value.shape),
            dtype=value.dtype,
            metadata={"dtype": str(self.dtype)},
            data={"value": value.to(self.dtype).contiguous()},
            stats=Stats(
                bytes_original=value.numel() * value.element_size(),
                bytes_compressed=value.numel() * self.dtype.itemsize,
            ),
        )
        return kp, vp

    def restore(self, key_payload: Payload, value_payload: Payload) -> tuple[torch.Tensor, torch.Tensor]:
        k = key_payload.data["value"].to(key_payload.dtype)
        v = value_payload.data["value"].to(value_payload.dtype)
        return k, v


class Float8Compressor(Compressor):
    """FP8 compressor class — re-exports :class:`kvfold.core.float8.Float8`."""

    method: str = "fp8"

    def __new__(cls, **kwargs: Any) -> Compressor:
        from kvfold.core.float8 import Float8
        return Float8(**kwargs)

    @classmethod
    def default_config(cls) -> Float8Config:
        return Float8Config()


def _register_builtins() -> None:
    from kvfold.core.jolt import Jolt
    from kvfold.core.flash import Flash
    from kvfold.core.low import Low
    from kvfold.core.identity import Pass
    from kvfold.core.float8 import Float8

    DISPATCH_REGISTRY.register("jolt", Jolt)
    DISPATCH_REGISTRY.register("flash", Flash)
    DISPATCH_REGISTRY.register("low", Low)
    DISPATCH_REGISTRY.register("pass", Pass)
    DISPATCH_REGISTRY.register("fp8", Float8)

    def int_factory(config: Any) -> Compressor:
        return IntQuant(
            bits=config.bits,
            symmetric=config.symmetric,
            per_channel=config.per_channel,
            group_size=config.group_size,
        )

    DISPATCH_REGISTRY.register("int2", IntQuant, factory=int_factory)
    DISPATCH_REGISTRY.register("int4", IntQuant, factory=int_factory)
    DISPATCH_REGISTRY.register("int8", IntQuant, factory=int_factory)

    def fp_factory(config: Any) -> Compressor:
        return FloatCast(dtype=config.dtype)

    DISPATCH_REGISTRY.register("fp16", FloatCast, factory=fp_factory)
    DISPATCH_REGISTRY.register("bf16", FloatCast, factory=fp_factory)


_register_builtins()
