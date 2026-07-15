"""JL-rotated residual path for JoLT.

After a partial Tucker truncation ``X̂ = partial_tucker(...)``, the residual
``R = X - X̂`` carries the energy the truncation discards. The paper
quantizes a *JL-rotated* version of ``R`` at low bit-width to recover most
of that energy cheaply.

Steps:

1. Compute ``R = X - X̂``.
2. Reshape ``R`` to a matrix with last axis ``dh``.
3. Apply a square JL projection ``Π``: ``R̃ = R @ Π.T``.
4. Quantize ``R̃`` uniformly at ``b`` bits.
5. Store ``Π``, the quantized codes, and the scale/zero-point.

Decode: invert the JL rotation, add the recovered residual to ``X̂``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from kvcompress.compressor.jl import JLDistribution, cached_projection
from kvcompress.compressor.quantization import (
    dequantize_tensor,
    get_quantizer,
    quantize_tensor,
)

log = logging.getLogger(__name__)


@dataclass
class ResidualPayload:
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
    projection_distribution: JLDistribution
    projection_sparsity: float
    quant_dtype: str
    symmetric: bool
    per_channel: bool
    group_size: int | None
    packed: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    original_shape: tuple[int, int, int]
    original_last: int

    @property
    def bits(self) -> int:
        return int(self.quant_dtype[3:])

    @property
    def bytes_compressed(self) -> int:
        bits = self.bits
        elem = self.packed.numel() * self.packed.element_size()
        bytes_per = (bits + 7) // 8
        # packed stores one byte per chunk of (8/bits) entries.
        n_entries = self.packed.numel() * (8 // max(bits, 1))
        return n_entries * bytes_per + elem + self.scale.numel() * 4 + self.zero_point.numel() * 4

    def to_dict(self) -> dict[str, object]:
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
    def from_dict(cls, d: dict[str, object]) -> "ResidualPayload":
        return cls(
            projection_seed=int(d["projection_seed"]),
            projection_distribution=d["projection_distribution"],  # type: ignore[arg-type]
            projection_sparsity=float(d["projection_sparsity"]),
            quant_dtype=str(d["quant_dtype"]),
            symmetric=bool(d["symmetric"]),
            per_channel=bool(d["per_channel"]),
            group_size=d.get("group_size"),  # type: ignore[arg-type]
            packed=d["packed"],  # type: ignore[arg-type]
            scale=d["scale"],  # type: ignore[arg-type]
            zero_point=d["zero_point"],  # type: ignore[arg-type]
            original_shape=tuple(d["original_shape"]),  # type: ignore[arg-type]
            original_last=int(d["original_last"]),
        )


def encode_residual(
    residual: torch.Tensor,
    *,
    bits: int,
    seed: int,
    distribution: JLDistribution = "gaussian",
    sparsity: float = 1.0,
    symmetric: bool = True,
    per_channel: bool = True,
    group_size: int | None = None,
) -> ResidualPayload:
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
        :class:`ResidualPayload`.
    """
    if bits not in (0, 2, 4, 8):
        raise ValueError(f"bits must be 0, 2, 4, or 8, got {bits}")
    if bits == 0:
        # No residual stored: encode an empty payload.
        return ResidualPayload(
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
    proj = cached_projection(
        output_dim=residual.shape[-1],
        input_dim=residual.shape[-1],
        distribution=distribution,
        seed=seed,
        device=residual.device,
        dtype=torch.float32,
        sparsity=sparsity,
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
    return ResidualPayload(
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


def decode_residual(payload: ResidualPayload, device: torch.device | None = None) -> torch.Tensor:
    """Decode a residual back to the original layout.

    Returns:
        Tensor of shape ``payload.original_shape``.
    """
    if payload.quant_dtype == "int0" or payload.packed.numel() == 0:
        return torch.zeros(payload.original_shape, dtype=torch.float32, device=device)

    dev = device or payload.packed.device
    proj = cached_projection(
        output_dim=payload.original_last,
        input_dim=payload.original_last,
        distribution=payload.projection_distribution,
        seed=payload.projection_seed,
        device=dev,
        dtype=torch.float32,
        sparsity=payload.projection_sparsity,
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


__all__ = [
    "ResidualPayload",
    "decode_residual",
    "encode_residual",
    "estimate_residual_bytes",
]