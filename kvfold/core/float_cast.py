"""Dtype-only halving compressor (fp16 / bf16).

Casts the input to a 16-bit float and stores it verbatim. The byte
count halves; the numerical content is unchanged. Use this when you
want to halve cache size without any algorithmic compression.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from kvfold.config import Fp16Config
from kvfold.core.base import Compressor, Payload, Stats

__all__ = ["FloatCast"]


log = logging.getLogger(__name__)


class FloatCast(Compressor):
    """Dtype-only compressor (fp16 / bf16).

    Attributes:
        dtype: target 16-bit float dtype.
    """

    method: str = "fp_cast"

    def __init__(self, *, dtype: torch.dtype = torch.float16, **unused: Any) -> None:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"FloatCast.dtype must be torch.float16 or torch.bfloat16; got {dtype}")
        if unused:
            from kvfold.errors import MethodConfigError
            raise MethodConfigError("fp_cast", ",".join(sorted(unused.keys())), "unknown field(s); accepts only dtype")
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
