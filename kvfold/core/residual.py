"""JL-rotated residual path for JoLT.

After a partial Tucker truncation ``X̂ = partial_tucker(...)``, the residual
``R = X - X̂`` carries the energy the truncation discards. The paper
quantizes a *JL-rotated* version of ``R`` at low bit-width to recover most
of that energy cheaply.

Pipeline:

1. Compute ``R = X - X̂``.
2. Reshape ``R`` to a matrix with last axis ``dh``.
3. Apply a square JL projection ``Π``: ``R̃ = R @ Π.T``.
4. Quantize ``R̃`` uniformly at ``b`` bits.
5. Store ``Π`` (seed only — it's regenerable), the quantized codes, and
   the scale/zero-point.

Decode: invert the JL rotation, add the recovered residual to ``X̂``.

Why JL before quantisation:

A residual's spectrum is *flat* (it's whatever the truncation missed).
Without rotation, the quantisation error budget would be spent
overwhelmingly on the largest components — the JL rotation spreads the
energy uniformly across all components, so a uniform b-bit quantiser
gives a roughly uniform b-bit error across all entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from kvfold.core.jl import Distribution, CACHE
from kvfold.core.quant import (
    dequantize_tensor,
    quantize_tensor,
)

__all__ = [
    "Residual",
    "decode_residual",
    "encode_residual",
    "estimate_residual_bytes",
]

log = logging.getLogger(__name__)


@dataclass
class Residual:
    """Serialized JL-rotated residual.

    Attributes:
        projection: cached JL projection matrix (kept for reproducibility,
            not stored long-term — only the seed matters).
        projection_seed: seed used to reconstruct the projection at decode.
        projection_distribution: ``"gaussian"`` or ``"rademacher"``.
        projection_sparsity: Achlioptas sparsity (1.0 for dense).
        quant_dtype: ``"int2"``, ``"int4"``, or ``"int8"``.
        symmetric: symmetric vs. asymmetric quantization.
        per_channel: per-channel scales.
        group_size: optional per-group size.
        packed: bit-packed codes (shape depends on sub-byte packing).
        scale: per-channel / per-group scale tensor.
        zero_point: zero point tensor.
        original_shape: shape of the residual before reshape.
        original_last: original last dim (for unpacking).
    """

    projection_seed: int
    projection_distribution: Distribution
    projection_sparsity: float
    quant_dtype: str
    symmetric: bool
    per_channel: bool
    group_size: int | None
    packed: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    original_shape: tuple[int, ...]
    original_last: int

    @property
    def bits(self) -> int:
        """Residual bit-width (``2``, ``4``, ``8``) extracted from ``quant_dtype``.

        ``quant_dtype="int0"`` is the "no residual" sentinel and returns
        ``0``.
        """
        return int(self.quant_dtype[3:])

    @property
    def bytes_compressed(self) -> int:
        """Exact byte cost of the residual payload.

        Counts the bit-packed storage exactly: ``packed.numel()`` bytes
        (uint8 container) plus the fp32 scale and zero-point tensors.
        The JL projection matrix is reconstructible from the seed so is
        not counted.
        """
        return (
            self.packed.numel() * self.packed.element_size()
            + self.scale.numel() * self.scale.element_size()
            + self.zero_point.numel() * self.zero_point.element_size()
        )

    def to_dict(self) -> dict[str, object]:
        """Convert this payload to a JSON-/safetensors-friendly dict.

        Tensors stay as tensors; lists/tuples become plain lists.
        """
        return {
            "projection_seed": self.projection_seed,
            "projection_distribution": self.projection_distribution,
            "projection_sparsity": self.projection_sparsity,
            "quant_dtype": self.quant_dtype,
            "symmetric": self.symmetric,
            "per_channel": self.per_channel,
            "group_size": self.group_size,
            "packed": self.packed,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "original_shape": list(self.original_shape),
            "original_last": self.original_last,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "Residual":
        """Inverse of :meth:`to_dict`. Recovers a payload from a dict.

        Raises:
            KeyError: if any required key is missing.
            TypeError: if a value's runtime type doesn't match the
                declared dataclass field.
        """
        return cls(
            projection_seed=int(d["projection_seed"]),  # type: ignore[arg-type,call-overload]
            projection_distribution=d["projection_distribution"],  # type: ignore[arg-type]
            projection_sparsity=float(d["projection_sparsity"]),  # type: ignore[arg-type]
            quant_dtype=str(d["quant_dtype"]),  # type: ignore[arg-type]
            symmetric=bool(d["symmetric"]),  # type: ignore[arg-type]
            per_channel=bool(d["per_channel"]),  # type: ignore[arg-type]
            group_size=d.get("group_size"),  # type: ignore[arg-type]
            packed=d["packed"],  # type: ignore[arg-type]
            scale=d["scale"],  # type: ignore[arg-type]
            zero_point=d["zero_point"],  # type: ignore[arg-type]
            original_shape=tuple(d["original_shape"]),  # type: ignore[arg-type]
            original_last=int(d["original_last"]),  # type: ignore[arg-type,call-overload]
        )


def encode_residual(
    residual: torch.Tensor,
    *,
    bits: int,
    seed: int,
    distribution: Distribution = "gaussian",
    sparsity: float = 1.0,
    symmetric: bool = True,
    per_channel: bool = True,
    group_size: int | None = None,
) -> Residual:
    """Encode a residual via JL rotation + uniform quantization.

    Args:
        residual: tensor of shape ``(m, T, dh)``.
        bits: 2, 4, or 8.
        seed: JL seed.
        distribution: ``"gaussian"`` or ``"rademacher"``.
        sparsity: Achlioptas sparsity for ``"rademacher"``.
        symmetric: symmetric or asymmetric quantization.
        per_channel: per-channel scales.
        group_size: optional per-group scale size.

    Returns:
        :class:`Residual`.
    """
    if bits not in (0, 2, 4, 8):
        raise ValueError(f"bits must be 0, 2, 4, or 8, got {bits}")
    if bits == 0:
        # No residual stored: encode an empty payload.
        return Residual(
            projection_seed=seed,
            projection_distribution=distribution,
            projection_sparsity=sparsity,
            quant_dtype="int0",
            symmetric=symmetric,
            per_channel=per_channel,
            group_size=group_size,
            packed=torch.zeros(0, dtype=torch.uint8, device=residual.device),
            scale=torch.zeros(0, device=residual.device),
            zero_point=torch.zeros(0, dtype=torch.int32, device=residual.device),
            original_shape=tuple(residual.shape),
            original_last=residual.shape[-1],
        )

    original_shape = tuple(residual.shape)
    m_total = residual.numel() // residual.shape[-1]

    # Reshape to (m_total, dh).
    flat = residual.reshape(m_total, residual.shape[-1]).to(torch.float32)

    # JL projection: dh x dh (square rotation).
    proj = CACHE.get_or_build(
        out_dim=residual.shape[-1],
        in_dim=residual.shape[-1],
        distribution=distribution,
        seed=seed,
        device=residual.device,
        dtype=torch.float32,
    )
    rotated = flat @ proj.matrix.t()

    quant_dtype = f"int{bits}"
    payload = quantize_tensor(
        rotated,
        dtype=quant_dtype,
        symmetric=symmetric,
        per_channel=per_channel,
        group_size=group_size,
    )
    return Residual(
        projection_seed=seed,
        projection_distribution=distribution,
        projection_sparsity=sparsity,
        quant_dtype=quant_dtype,
        symmetric=symmetric,
        per_channel=per_channel,
        group_size=group_size,
        packed=payload["q"],
        scale=payload["scale"],
        zero_point=payload["zero_point"],
        original_shape=original_shape,
        original_last=int(payload["original_last"].item()),
    )


def decode_residual(payload: Residual, device: torch.device | None = None) -> torch.Tensor:
    """Decode a residual back to the original layout.

    Steps:
        1. If the payload is the ``int0`` sentinel, return zeros of
           ``payload.original_shape`` (no residual was stored).
        2. Reconstruct the JL projection from the seed.
        3. Dequantise the rotated codes.
        4. Apply the inverse JL rotation: ``R = rotated @ Π``.
        5. Reshape to ``payload.original_shape``.

    Args:
        payload: the residual payload.
        device: override device for the decoded tensor. Defaults to
            ``payload.packed.device``.

    Returns:
        Tensor of shape ``payload.original_shape``, dtype ``fp32``.
    """
    if payload.quant_dtype == "int0" or payload.packed.numel() == 0:
        return torch.zeros(payload.original_shape, dtype=torch.float32, device=device)

    dev = device or payload.packed.device
    proj = CACHE.get_or_build(
        out_dim=payload.original_last,
        in_dim=payload.original_last,
        distribution=payload.projection_distribution,
        seed=payload.projection_seed,
        device=dev,
        dtype=torch.float32,
    )
    rotated = dequantize_tensor(
        {
            "q": payload.packed.to(dev),
            "scale": payload.scale.to(dev),
            "zero_point": payload.zero_point.to(dev),
            "original_last": torch.tensor(payload.original_last, dtype=torch.int32),
        },
        dtype=payload.quant_dtype,
        symmetric=payload.symmetric,
        per_channel=payload.per_channel,
        group_size=payload.group_size,
        output_dtype=torch.float32,
    )
    flat = rotated @ proj.matrix  # apply_inverse
    return flat.reshape(payload.original_shape)


def estimate_residual_bytes(
    shape: tuple[int, int, int],
    bits: int,
    per_channel: bool = True,
    group_size: int | None = None,
) -> int:
    """Bytes occupied by a residual payload of given shape and bit-width.

    Used by the allocator. Includes the JL projection matrix at fp32, the
    packed quantized codes, and per-channel / per-group scales at fp32.
    """
    m, t, d = shape
    if bits == 0:
        return 0
    n = m * t * d
    packed_bytes = (n * bits + 7) // 8
    proj_bytes = d * d * 4  # fp32
    if per_channel:
        scale_bytes = m * t * 4 + m * t * 4  # scale + zero_point
    elif group_size is not None:
        n_groups = (m * t * d) // group_size
        scale_bytes = n_groups * 8
    else:
        scale_bytes = 8
    return packed_bytes + proj_bytes + scale_bytes
