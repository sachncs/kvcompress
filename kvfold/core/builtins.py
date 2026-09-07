"""Concrete :class:`kvfold.core.base.Compressor` subclasses.

Each entry in :data:`REGISTRY` is bound at module import time via
``_register_builtins()``. The actual class implementations live in their
own modules; this file just wires them into the dispatcher.

Methods registered here:

* ``jolt`` -> :class:`Jolt`
* ``flash`` -> :class:`Flash`
* ``low`` -> :class:`Low`
* ``pass`` -> :class:`Pass`
* ``fp8`` -> :class:`Float8`
* ``int2``, ``int4``, ``int8`` -> :class:`IntQuant` (factory sets bits)
* ``fp16``, ``bf16`` -> :class:`FloatCast` (factory sets dtype)
"""

from __future__ import annotations

from typing import Any

from kvfold.config import (
    Bf16Config,
    FlashConfig,
    Float8Config,
    FloatConfig,
    Fp16Config,
    Int2Config,
    Int4Config,
    Int8Config,
    IntConfig,
    JoltConfig,
    LowRankConfig,
    PassConfig,
)
from kvfold.core.base import Compressor, Payload, Stats
from kvfold.core.dispatch import REGISTRY as DISPATCH_REGISTRY

__all__ = ["DISPATCH_REGISTRY"]


def _register_builtins() -> None:
    from kvfold.core.jolt import Jolt
    from kvfold.core.flash import Flash
    from kvfold.core.low import Low
    from kvfold.core.identity import Pass
    from kvfold.core.float8 import Float8
    from kvfold.core.int_quant import IntQuant as IntQuantCompressor
    from kvfold.core.float_cast import FloatCast

    DISPATCH_REGISTRY.register("jolt", Jolt)
    DISPATCH_REGISTRY.register("flash", Flash)
    DISPATCH_REGISTRY.register("low", Low)
    DISPATCH_REGISTRY.register("pass", Pass)
    DISPATCH_REGISTRY.register("fp8", Float8)

    def int_factory(config: Any) -> Compressor:
        return IntQuantCompressor(
            bits=config.bits,
            symmetric=config.symmetric,
            per_channel=config.per_channel,
            group_size=config.group_size,
        )

    DISPATCH_REGISTRY.register("int2", IntQuantCompressor, factory=int_factory)
    DISPATCH_REGISTRY.register("int4", IntQuantCompressor, factory=int_factory)
    DISPATCH_REGISTRY.register("int8", IntQuantCompressor, factory=int_factory)

    def fp_factory(config: Any) -> Compressor:
        return FloatCast(dtype=config.dtype)

    DISPATCH_REGISTRY.register("fp16", FloatCast, factory=fp_factory)
    DISPATCH_REGISTRY.register("bf16", FloatCast, factory=fp_factory)


_register_builtins()
