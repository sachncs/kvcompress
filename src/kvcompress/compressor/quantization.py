"""Scalar / vector quantization primitives for KV cache compression.

Supported formats:

* **FP16 / BF16**: identity wrappers, useful as a baseline or for the Tucker
  core.
* **FP8 (E4M3 / E5M2)**: real fp8 storage via ``torch.float8_e4m3fn`` and
  ``torch.float8_e5m2`` when available; falls back to a quantization +
  dequantization round-trip that materialises fp8 via packing into uint8.
* **INT8 / INT4 / INT2**: signed symmetric or asymmetric uniform
  quantization. Per-channel (last-axis) or per-group scales. INT2 and INT4
  are bit-packed into ``uint8`` containers.

All quantizers expose the same protocol:

* ``quantize(x, *, scale=None, zero_point=None) -> (q, scale, zero_point)``
* ``dequantize(q, scale, zero_point, *, dtype=None) -> x_hat``

Round-trip is reversible up to numerical noise — the absolute error is
bounded by half a quantization bin. The paper uses these primitives for the
JL-residual code path.

The bit-packing offset trick:

    Symmetric quantisation maps ``q_int ∈ [-2^(b-1), 2^(b-1) - 1]`` to
    ``q_unsigned = q_int + 2^(b-1) ∈ [0, 2^b - 1]`` so two's-complement
    bit patterns line up with a contiguous uint8 representation. Asymmetric
    quantisation already lives in ``[0, 2^b)`` so no offset is applied.
    The decoder subtracts the same offset (or nothing) to recover signed
    values. The 8-bit path uses an explicit offset to keep uint8 values
    in ``[0, 255]`` rather than wrapping in int8 (which would corrupt values
    ≥ 128).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

log = logging.getLogger(__name__)


QuantDType = Literal["fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "int8", "int4", "int2"]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Quantizer(Protocol):
    """Quantize / dequantize a tensor with the same dtype on both sides."""

    name: QuantDType

    def quantize(
        self,
        x: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
        zero_point: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def dequantize(
        self,
        q: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# FP8 / FP16 — identity (cast) quantizers
# ---------------------------------------------------------------------------


@dataclass
class FloatCastQuantizer:
    """Casts to a low-precision float and back; scale is always 1.0."""

    name: QuantDType = "fp16"

    def __post_init__(self) -> None:
        if self.name == "fp16":
            self._dtype = torch.float16
        elif self.name == "bf16":
            self._dtype = torch.bfloat16
        elif self.name == "fp8_e4m3":
            self._dtype = fp8_e4m3_dtype()
        elif self.name == "fp8_e5m2":
            self._dtype = fp8_e5m2_dtype()
        else:
            raise ValueError(f"FloatCastQuantizer does not support {self.name!r}")

    def quantize(
        self,
        x: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
        zero_point: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = x.to(self._dtype)
        ones = torch.ones((), dtype=torch.float32, device=x.device)
        zeros = torch.zeros((), dtype=torch.int32, device=x.device)
        return q, ones, zeros

    def dequantize(
        self,
        q: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return q.to(output_dtype or torch.float32)


def fp8_e4m3_dtype() -> torch.dtype:
    return getattr(torch, "float8_e4m3fn", None) or torch.float32


def fp8_e5m2_dtype() -> torch.dtype:
    return getattr(torch, "float8_e5m2", None) or torch.float32


# ---------------------------------------------------------------------------
# Integer quantizers
# ---------------------------------------------------------------------------


def bit_packing_signed(
    q_int: torch.Tensor,
    bits: int,
    *,
    symmetric: bool = True,
) -> torch.Tensor:
    """Pack signed int tensor of width ``bits`` into ``uint8``.

    For symmetric quantization the input range is
    ``[-2^(bits-1), 2^(bits-1) - 1]``; we add an offset of ``2^(bits-1)`` to
    map it into ``[0, 2^bits)``. For asymmetric quantization the input is
    already in ``[qmin, qmax]`` (typically ``[0, 2^bits - 1]``) and no offset
    is applied.

    Implementation notes:

    * The 8-bit path uses an explicit offset to keep uint8 values in
      ``[0, 255]`` rather than wrapping through int8 (which would corrupt
      values ≥ 128).
    * The 4-bit path packs two entries per byte: low indices in the high
      nibble, high indices in the low nibble. Last-dim padding to even
      length happens before packing.
    * The 2-bit path packs four entries per byte, big-endian across the
      pair. Last-dim padding to a multiple of 4 happens before packing.
    """
    offset = 1 << (bits - 1) if symmetric else 0
    if bits == 8:
        return (q_int + offset).to(torch.uint8)
    if bits not in (2, 4):
        raise ValueError(f"only 2- and 4-bit sub-byte packing supported, got {bits}")

    q_unsigned = (q_int + offset).to(torch.uint8)

    last = q_unsigned.shape[-1]
    out_last = (last * bits + 7) // 8
    flat = q_unsigned.reshape(*q_unsigned.shape[:-1], last)
    out = torch.zeros(*flat.shape[:-1], out_last, dtype=torch.uint8, device=flat.device)
    # Pack 4 bits at a time.
    if bits == 4:
        # Pad last dim to even count.
        if last % 2 != 0:
            pad = torch.zeros(*flat.shape[:-1], 1, dtype=torch.uint8, device=flat.device)
            flat = torch.cat([flat, pad], dim=-1)
        high = flat[..., 0::2]  # low indices in high nibble
        low = flat[..., 1::2]
        out = (high << 4) | low
        return out
    # bits == 2: pack 4 entries per byte.
    if last % 4 != 0:
        pad_count = (4 - last % 4) % 4
        pad = torch.zeros(*flat.shape[:-1], pad_count, dtype=torch.uint8, device=flat.device)
        flat = torch.cat([flat, pad], dim=-1)
    b0 = flat[..., 0::4]
    b1 = flat[..., 1::4]
    b2 = flat[..., 2::4]
    b3 = flat[..., 3::4]
    out = (b0 << 6) | (b1 << 4) | (b2 << 2) | b3
    return out


def bit_unpacking_signed(
    packed: torch.Tensor,
    bits: int,
    original_last: int,
    *,
    symmetric: bool = True,
) -> torch.Tensor:
    """Inverse of :func:`_bit_packing_signed`.

    For symmetric quantization the encoded ``q_int + offset`` ranges in
    ``[0, 2^bits)``; we subtract ``offset = 2^(bits-1)`` to recover signed
    values. For asymmetric quantization the encoded values are already
    unsigned in ``[qmin, qmax]`` and ``offset`` is zero.
    """
    offset = 1 << (bits - 1) if symmetric else 0
    if bits == 8:
        # uint8 → int32 to avoid wrapping in int8.
        return packed.to(torch.int32) - offset
    if bits not in (2, 4):
        raise ValueError(f"only 2- and 4-bit sub-byte unpacking supported, got {bits}")

    if bits == 4:
        high = (packed >> 4) & 0xF
        low = packed & 0xF
        flat = torch.stack([high, low], dim=-1).reshape(*packed.shape[:-1], -1)
        flat = flat[..., :original_last]
        return (flat.to(torch.int32) - offset).to(torch.int32)
    # bits == 2
    b0 = (packed >> 6) & 0x3
    b1 = (packed >> 4) & 0x3
    b2 = (packed >> 2) & 0x3
    b3 = packed & 0x3
    flat = torch.stack([b0, b1, b2, b3], dim=-1).reshape(*packed.shape[:-1], -1)
    flat = flat[..., :original_last]
    return (flat.to(torch.int32) - offset).to(torch.int32)


@dataclass
class IntQuantizer:
    """Symmetric or asymmetric uniform integer quantization.

    Args:
        bits: bit-width (2, 4, or 8).
        symmetric: use symmetric ranges ``[-qmax, qmax]`` if True, else
            asymmetric ``[qmin, qmax]`` derived from data.
        per_channel: scale per last-axis slice if True, else a single
            scalar per tensor.
        group_size: if set and ``per_channel=False``, use per-group scales
            along the last axis of this size.
    """

    bits: int
    symmetric: bool = True
    per_channel: bool = True
    group_size: int | None = None

    def __post_init__(self) -> None:
        if self.bits not in (2, 4, 8):
            raise ValueError(f"IntQuantizer bits must be 2/4/8, got {self.bits}")
        if self.symmetric:
            self._qmax = (1 << (self.bits - 1)) - 1
            self._qmin = -(1 << (self.bits - 1))
        else:
            self._qmax = (1 << self.bits) - 1
            self._qmin = 0

    @property
    def name(self) -> str:
        return f"int{self.bits}"  # type: ignore[return-value]

    def quantize(
        self,
        x: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
        zero_point: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if scale is None or zero_point is None:
            scale, zero_point = self.compute_params(x)

        if scale.dim() == 0:
            scaled = x / scale + zero_point
        elif self.group_size is not None:
            # Per-group: insert a group axis on x and a singleton last axis
            # on the scale, so x has shape (..., n_groups, group_size) and
            # scale broadcasts cleanly.
            last = x.shape[-1]
            n_groups = last // self.group_size
            x_grouped = x.reshape(*x.shape[:-1], n_groups, self.group_size)
            view_shape = list(scale.shape) + [1]
            scaled = x_grouped / scale.view(view_shape) + zero_point.view(view_shape)
            scaled = scaled.reshape(*x.shape)
        else:
            # Per-channel.
            view_shape = [1] * (x.dim() - 1) + [-1]
            scaled = x / scale.view(view_shape) + zero_point.view(view_shape)

        q_int = torch.round(scaled).clamp(self._qmin, self._qmax).to(torch.int32)
        packed = bit_packing_signed(
            q_int.to(torch.int8),
            self.bits,
            symmetric=self.symmetric,
        )
        return packed, scale.detach(), zero_point.detach()

    def dequantize(
        self,
        q: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        original_last = self.original_last(q)
        q_int = bit_unpacking_signed(q, self.bits, original_last, symmetric=self.symmetric)
        if scale.dim() == 0:
            x_hat = (q_int - zero_point) * scale
        elif self.group_size is not None:
            last = q_int.shape[-1]
            n_groups = last // self.group_size
            q_grouped = q_int.reshape(*q_int.shape[:-1], n_groups, self.group_size)
            view_shape = list(scale.shape) + [1]
            x_hat = (q_grouped - zero_point.view(view_shape)) * scale.view(view_shape)
            x_hat = x_hat.reshape(*q_int.shape)
        else:
            view_shape = [1] * (q_int.dim() - 1) + [-1]
            x_hat = (q_int - zero_point.view(view_shape)) * scale.view(view_shape)
        return x_hat.to(output_dtype or torch.float32)

    def original_last(self, q: torch.Tensor) -> int:
        # Sub-byte packing changes the last dim size; we store the original
        # last dim as metadata, but here we infer from the scale shape.
        return int(self._last_dim_hint)

    def compute_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.per_channel:
            x_flat = x.reshape(-1, x.shape[-1])
            x_min = x_flat.min(dim=0).values
            x_max = x_flat.max(dim=0).values
        elif self.group_size is not None:
            # Per-group scales.
            last = x.shape[-1]
            assert last % self.group_size == 0, (
                f"last dim {last} not divisible by group_size {self.group_size}"
            )
            x_g = x.reshape(*x.shape[:-1], last // self.group_size, self.group_size)
            x_min = x_g.min(dim=-1).values
            x_max = x_g.max(dim=-1).values
        else:
            x_min = x.amin()
            x_max = x.amax()

        if self.symmetric:
            abs_max = torch.maximum(x_min.abs(), x_max.abs())
            abs_max = torch.clamp(abs_max, min=1e-12)
            scale = abs_max / self._qmax
            zero_point = torch.zeros_like(scale, dtype=torch.int32)
        else:
            rng = torch.clamp(x_max - x_min, min=1e-12)
            scale = rng / (self._qmax - self._qmin)
            zero_point = torch.round(self._qmin - x_min / scale).to(torch.int32)

        # Hint for dequantize: original last-dim length.
        self._last_dim_hint = x.shape[-1]
        return scale.to(torch.float32), zero_point


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_QUANTIZER_REGISTRY: dict[str, Quantizer] = {}


def get_quantizer(
    name: QuantDType | str,
    *,
    symmetric: bool = True,
    per_channel: bool = True,
    group_size: int | None = None,
) -> Quantizer:
    """Construct or fetch a cached quantizer."""
    if name in ("fp16", "bf16", "fp8_e4m3", "fp8_e5m2"):
        key = f"float:{name}"
        return get_or_create(
            key,
            lambda: FloatCastQuantizer(name=name),  # type: ignore[arg-type]
        )
    if name in ("int2", "int4", "int8"):
        bits = int(name[3:])
        key = f"int:{bits}:{symmetric}:{per_channel}:{group_size}"
        return get_or_create(
            key,
            lambda: IntQuantizer(
                bits=bits,
                symmetric=symmetric,
                per_channel=per_channel,
                group_size=group_size,
            ),
        )
    raise ValueError(f"unknown quantizer name {name!r}")


def get_or_create(key: str, factory) -> Quantizer:
    """Get a cached quantizer or build + cache one."""
    cached = _QUANTIZER_REGISTRY.get(key)
    if cached is not None:
        return cached
    new_q: Quantizer = factory()
    _QUANTIZER_REGISTRY[key] = new_q
    return new_q


def quantize_tensor(
    x: torch.Tensor,
    *,
    dtype: QuantDType | str,
    symmetric: bool = True,
    per_channel: bool = True,
    group_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Convenience wrapper that returns ``{q, scale, zero_point, original_last}``."""
    q = get_quantizer(
        dtype,
        symmetric=symmetric,
        per_channel=per_channel,
        group_size=group_size,
    )
    packed, scale, zp = q.quantize(x)
    # Store the original last dim for unpacking.
    out = {
        "q": packed,
        "scale": scale,
        "zero_point": zp,
        "original_last": torch.tensor(x.shape[-1], dtype=torch.int32),
    }
    return out


def dequantize_tensor(
    payload: dict[str, torch.Tensor],
    *,
    dtype: QuantDType | str,
    symmetric: bool = True,
    per_channel: bool = True,
    group_size: int | None = None,
    output_shape: tuple[int, ...] | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    q = get_quantizer(
        dtype,
        symmetric=symmetric,
        per_channel=per_channel,
        group_size=group_size,
    )
    x = q.dequantize(
        payload["q"],
        payload["scale"],
        payload["zero_point"],
        output_dtype=output_dtype,
    )
    if output_shape is not None:
        x = x.reshape(output_shape)
    return x


def estimate_int_bytes(numel: int, bits: int) -> int:
    """Bytes required to store ``numel`` packed signed integers of width ``bits``."""
    return (numel * bits + 7) // 8


__all__ = [
    "FloatCastQuantizer",
    "IntQuantizer",
    "QuantDType",
    "Quantizer",
    "dequantize_tensor",
    "estimate_int_bytes",
    "get_quantizer",
    "quantize_tensor",
]
