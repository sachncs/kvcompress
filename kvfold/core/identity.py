"""Passthrough compressor — stores K/V with optional dtype cast, no real compression.

:class:`Pass` writes the input tensor (cast to ``dtype``) into the payload
verbatim. The bytes_compressed figure reflects the cast dtype, so a
fp16 cast halves the on-disk size; this is the same effect as casting
the cache to fp16 globally. There is no algorithmic compression.

For real compression use :class:`Jolt`, :class:`Flash`, :class:`Low`, or
:class:`IntQuant`. For dtype-only halving use :class:`FloatCast` directly.

The kwarg is named ``dtype`` (was ``factor_dtype`` in the previous
revision). Other compressors use ``dtype`` to mean the same thing.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from kvfold.config import PassConfig
from kvfold.core.base import Compressor, Payload, Stats

__all__ = ["Pass"]

log = logging.getLogger(__name__)


class Pass(Compressor):
    """Passthrough with optional dtype cast.

    The name reflects semantics: nothing is changed beyond what the
    storage dtype allows. This is a useful baseline for ablation studies
    and for "disable compression cleanly" code paths.

    Attributes:
        dtype: storage dtype; tensors are cast to this on put.
    """

    method: str = "pass"

    def __init__(self, *, dtype: torch.dtype | None = None, **unused: Any) -> None:
        self.dtype = dtype if dtype is not None else torch.float16

    @classmethod
    def default_config(cls) -> PassConfig:
        return PassConfig()

    def compress(self, key: torch.Tensor, value: torch.Tensor) -> tuple[Payload, Payload]:
        """Store K and V verbatim (modulo a single dtype cast)."""
        Compressor.validate(key, value)
        kp = Payload(
            method="pass",
            shape=tuple(key.shape),
            dtype=key.dtype,
            metadata={"r_token": 0, "r_feature": 0, "bits": 0},
            data={"value": key.to(self.dtype).contiguous()},
            stats=Stats(
                bytes_original=key.numel() * key.element_size(),
                bytes_compressed=key.numel() * self.dtype.itemsize,
            ),
        )
        vp = Payload(
            method="pass",
            shape=tuple(value.shape),
            dtype=value.dtype,
            metadata={"r_token": 0, "r_feature": 0, "bits": 0},
            data={"value": value.to(self.dtype).contiguous()},
            stats=Stats(
                bytes_original=value.numel() * value.element_size(),
                bytes_compressed=value.numel() * self.dtype.itemsize,
            ),
        )
        return kp, vp

    def restore(self, key_payload: Payload, value_payload: Payload) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct K and V from the cached buffers."""
        k = key_payload.data["value"].to(key_payload.dtype)
        v = value_payload.data["value"].to(value_payload.dtype)
        return k, v
