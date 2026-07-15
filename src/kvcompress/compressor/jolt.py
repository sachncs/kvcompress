"""JoLT compressor — paper-faithful implementation.

Combines:

* partial Tucker decomposition (token + feature modes),
* JL-rotated low-bit residual,
* per-(layer group, K/V) Lagrangian allocator,
* ST-HOSVD with the head+layer modes pinned.

The compressor takes K and V at one layer, applies the allocator to decide
ranks and residual bit-widths jointly for K and V, compresses each, and
returns two :class:`~kvcompress.compressor.base.CompressedPayload`
objects ready for storage.

Algorithm (per compress() call):

1. Build two :class:`~kvcompress.compressor.allocator.Cell` instances (one
   each for K and V) describing the cell shape and budget knobs.
2. Call :meth:`JointAllocator.optimize` to get the per-cell
   ``(r_token, r_feature, bits)`` decisions.
3. For each cell: run ST-HOSVD via
   :func:`~kvcompress.compressor.tucker.partial_tucker_st_hosvd`,
   compute the residual, JL-rotate and quantise it via
   :func:`~kvcompress.compressor.residual.encode_residual`.
4. Package the core + bases + residual payload into a
   :class:`CompressedPayload` and return the K and V payloads.

The :meth:`decompress` method is the exact inverse.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch

from kvcompress.compressor.allocator import (
    AllocationResult,
    Cell,
    JointAllocator,
)
from kvcompress.compressor.base import (
    CompressedPayload,
    CompressorStats,
    KVCompressor,
)
from kvcompress.compressor.residual import (
    ResidualPayload,
    decode_residual,
    encode_residual,
)
from kvcompress.compressor.svd import SVD
from kvcompress.compressor.tucker import (
    TuckerFactors,
    partial_tucker_st_hosvd,
    reconstruct_partial_tucker,
)

log = logging.getLogger(__name__)


@dataclass
class _JoLTFactors:
    """Internal per-cell (key or value) JoLT factors.

    Attributes:
        tucker: partial Tucker decomposition result.
        residual: encoded JL-rotated residual.
        allocation: allocator decision for this cell.
    """

    tucker: TuckerFactors
    residual: ResidualPayload | None
    allocation: Any


class JoLTCompressor(KVCompressor):
    """Paper-faithful JoLT compressor.

    Args:
        compression_ratio: target compression ratio (e.g. ``3.0``).
        bits: residual bit-widths the allocator can choose from.
        factor_dtype: dtype of stored Tucker factors (``fp16`` or ``fp32``).
        jl_distribution: ``"gaussian"`` or ``"rademacher"``.
        allocator: optional pre-built :class:`JointAllocator`. If ``None``,
            one is constructed from ``compression_ratio`` and ``bits``.
        svd: optional shared :class:`SVD` (for deterministic seeding).
        symmetric_quant: symmetric vs. asymmetric quantization.
        per_channel_quant: per-channel scales.
        group_size: per-group scale size (only if not per_channel).
        layer_groups: number of layer groups (kept at 1 for compatibility;
            the paper uses G=1).
    """

    name = "jolt"

    def __init__(
        self,
        *,
        compression_ratio: float = 3.0,
        bits: tuple[int, ...] = (0, 2, 4, 8),
        factor_dtype: torch.dtype = torch.float16,
        jl_distribution: str = "gaussian",
        allocator: JointAllocator | None = None,
        svd: SVD | None = None,
        symmetric_quant: bool = True,
        per_channel_quant: bool = True,
        group_size: int | None = None,
        layer_groups: int = 1,
        seed: int = 0,
        **_unused: Any,
    ) -> None:
        super().__init__()
        if compression_ratio <= 1.0:
            raise ValueError(f"compression_ratio must be > 1.0, got {compression_ratio}")
        self.compression_ratio = float(compression_ratio)
        self.bits = tuple(bits)
        self.factor_dtype = factor_dtype
        self.jl_distribution = jl_distribution
        self.allocator = allocator or JointAllocator(
            target_ratio=compression_ratio,
            bits_grid=self.bits,
        )
        self.svd = svd or SVD(seed=seed, method="exact")
        self.symmetric_quant = symmetric_quant
        self.per_channel_quant = per_channel_quant
        self.group_size = group_size
        self.layer_groups = int(layer_groups)
        self.seed = int(seed)
        self._last_stats = CompressorStats()
        self._call_count = 0

    # ------------------------------------------------------------------
    # Compress / decompress
    # ------------------------------------------------------------------

    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[CompressedPayload, CompressedPayload]:
        if key.shape != value.shape:
            raise ValueError(f"K/V shape mismatch: {key.shape} vs {value.shape}")
        if key.dim() != 3:
            raise ValueError(f"JoLT expects 3-D (m, T, dh); got {tuple(key.shape)}")

        t0 = time.perf_counter()
        m, t, d = key.shape
        # Build allocator cells for K and V together.
        cells = [
            Cell(
                shape=(m, t, d),
                kind="key",
                layer_group=0,
                candidate_bits=self.bits,
            ),
            Cell(
                shape=(m, t, d),
                kind="value",
                layer_group=0,
                candidate_bits=self.bits,
            ),
        ]
        alloc_result: AllocationResult = self.allocator.optimize(cells)

        k_alloc, v_alloc = alloc_result.allocations

        k_factors = self.compress_cell(key, k_alloc)
        v_factors = self.compress_cell(value, v_alloc)

        k_payload = self.build_payload(key, k_factors)
        v_payload = self.build_payload(value, v_factors)

        elapsed = (time.perf_counter() - t0) * 1000
        self._last_stats = CompressorStats(
            compress_time_ms=elapsed,
            bytes_original=key.numel() * key.element_size() * 2,
            bytes_compressed=k_payload.bytes_compressed + v_payload.bytes_compressed,
            extra={
                "k_r_token": k_factors.allocation.r_token,
                "k_r_feature": k_factors.allocation.r_feature,
                "k_bits": k_factors.allocation.bits,
                "v_r_token": v_factors.allocation.r_token,
                "v_r_feature": v_factors.allocation.r_feature,
                "v_bits": v_factors.allocation.bits,
                "lambda_star": alloc_result.lambda_star,
                "achieved_ratio": alloc_result.achieved_ratio,
                "target_ratio": alloc_result.target_ratio,
            },
        )
        self._call_count += 1
        return k_payload, v_payload

    def decompress(
        self,
        key_payload: CompressedPayload,
        value_payload: CompressedPayload,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t0 = time.perf_counter()
        k = self.reconstruct_payload(key_payload)
        v = self.reconstruct_payload(value_payload)
        elapsed = (time.perf_counter() - t0) * 1000
        # Update stats; don't overwrite compress stats.
        if self._last_stats.decompress_time_ms == 0.0:
            self._last_stats.decompress_time_ms = elapsed
        return k, v

    def stats(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "call_count": self._call_count,
            **self._last_stats.to_dict(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def compress_cell(
        self,
        x: torch.Tensor,
        allocation: Any,
    ) -> _JoLTFactors:
        rt = int(allocation.r_token)
        rd = int(allocation.r_feature)
        b = int(allocation.bits)
        # ST-HOSVD
        tucker = partial_tucker_st_hosvd(
            x,
            r_token=rt,
            r_feature=rd,
            svd=self.svd,
        )
        # Residual
        recon = reconstruct_partial_tucker(tucker, x.shape)
        residual_tensor = (x - recon).contiguous()
        if b == 0:
            residual: ResidualPayload | None = encode_residual(
                residual_tensor,
                bits=0,
                seed=self.seed,
                distribution=self.jl_distribution,  # type: ignore[arg-type]
            )
        else:
            residual = encode_residual(
                residual_tensor,
                bits=b,
                seed=self.seed,
                distribution=self.jl_distribution,  # type: ignore[arg-type]
                symmetric=self.symmetric_quant,
                per_channel=self.per_channel_quant,
                group_size=self.group_size,
            )
        return _JoLTFactors(tucker=tucker, residual=residual, allocation=allocation)

    def build_payload(self, original: torch.Tensor, factors: _JoLTFactors) -> CompressedPayload:
        # Cast factors to storage dtype.
        core = factors.tucker.core.to(self.factor_dtype).contiguous()
        u_token = factors.tucker.u_token.to(self.factor_dtype).contiguous()
        u_feature = factors.tucker.u_feature.to(self.factor_dtype).contiguous()

        data: dict[str, Any] = {
            "core": core,
            "u_token": u_token,
            "u_feature": u_feature,
        }
        meta: dict[str, Any] = {
            "method": "jolt",
            "r_token": factors.tucker.r_token,
            "r_feature": factors.tucker.r_feature,
            "bits": factors.allocation.bits,
            "core_dtype": "fp16" if self.factor_dtype == torch.float16 else "fp32",
            "tail_token_mass": factors.tucker.token_tail_mass,
            "tail_feature_mass": factors.tucker.feature_tail_mass,
        }
        if factors.residual is not None:
            meta["residual_seed"] = factors.residual.projection_seed
            meta["residual_distribution"] = factors.residual.projection_distribution
            meta["residual_sparsity"] = factors.residual.projection_sparsity
            meta["residual_symmetric"] = factors.residual.symmetric
            meta["residual_per_channel"] = factors.residual.per_channel
            meta["residual_group_size"] = factors.residual.group_size
            data["residual_packed"] = factors.residual.packed
            data["residual_scale"] = factors.residual.scale
            data["residual_zero_point"] = factors.residual.zero_point
            data["residual_original_shape"] = torch.tensor(
                list(factors.residual.original_shape), dtype=torch.int32
            )
            data["residual_original_last"] = torch.tensor(
                factors.residual.original_last, dtype=torch.int32
            )
            data["residual_dtype"] = torch.tensor(
                int(factors.residual.quant_dtype[3:]), dtype=torch.int32
            )

        return CompressedPayload(
            method="jolt",
            shape=tuple(original.shape),
            dtype=original.dtype,
            metadata=meta,
            data=data,
            stats=CompressorStats(
                bytes_original=original.numel() * original.element_size(),
                bytes_compressed=core.numel() * core.element_size()
                + u_token.numel() * u_token.element_size()
                + u_feature.numel() * u_feature.element_size(),
            ),
        )

    def reconstruct_payload(self, payload: CompressedPayload) -> torch.Tensor:
        if payload.method != "jolt":
            raise ValueError(f"JoLT cannot decode payload with method={payload.method!r}")
        core = payload.data["core"].to(torch.float32)
        u_token = payload.data["u_token"].to(torch.float32)
        u_feature = payload.data["u_feature"].to(torch.float32)
        x = reconstruct_partial_tucker(
            TuckerFactors(
                core=core,
                u_token=u_token,
                u_feature=u_feature,
                token_sv=torch.empty(0),
                feature_sv=torch.empty(0),
                token_tail_mass=float(payload.metadata.get("tail_token_mass", 0.0)),
                feature_tail_mass=float(payload.metadata.get("tail_feature_mass", 0.0)),
            ),
            target_shape=payload.shape,  # type: ignore[arg-type]
        )

        # Add residual if present.
        if "residual_packed" in payload.data:
            res_dtype_int = int(payload.data["residual_dtype"].item())
            quant_dtype = f"int{res_dtype_int}" if res_dtype_int > 0 else "int0"
            original_shape = tuple(int(d) for d in payload.data["residual_original_shape"].tolist())
            original_last = int(payload.data["residual_original_last"].item())
            residual = ResidualPayload(
                projection_seed=int(payload.metadata["residual_seed"]),
                projection_distribution=payload.metadata["residual_distribution"],  # type: ignore[arg-type]
                projection_sparsity=float(payload.metadata["residual_sparsity"]),
                quant_dtype=quant_dtype,
                symmetric=bool(payload.metadata["residual_symmetric"]),
                per_channel=bool(payload.metadata["residual_per_channel"]),
                group_size=payload.metadata.get("residual_group_size"),
                packed=payload.data["residual_packed"],
                scale=payload.data["residual_scale"],
                zero_point=payload.data["residual_zero_point"],
                original_shape=original_shape,
                original_last=original_last,
            )
            recovered = decode_residual(residual)
            x = x + recovered

        return x.to(payload.dtype)


__all__ = ["JoLTCompressor"]
