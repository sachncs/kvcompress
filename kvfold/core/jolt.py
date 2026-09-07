"""JoLT compressor — paper-faithful implementation.

Combines:

* partial Tucker decomposition (token + feature modes),
* JL-rotated low-bit residual,
* per-(layer group, K/V) Lagrangian allocator,
* ST-HOSVD with the head+layer modes pinned.

The compressor takes K and V at one layer, applies the allocator to decide
ranks and residual bit-widths jointly for K and V, compresses each, and
returns two :class:`~kvfold.core.base.Payload`
objects ready for storage.

Algorithm (per compress() call):

1. Build two :class:`~kvfold.core.allocator.Cell` instances (one
   each for K and V) describing the cell shape and budget knobs.
2. Call :meth:`Bisect.optimize` to get the per-cell
   ``(r_token, r_feature, bits)`` decisions.
3. For each cell: run ST-HOSVD via
   :func:`~kvfold.core.tucker.partial_tucker_st_hosvd`,
   compute the residual, JL-rotate and quantise it via
   :func:`~kvfold.core.residual.encode_residual`.
4. Package the core + bases + residual payload into a
   :class:`Payload` and return the K and V payloads.

The :meth:`decompress` method is the exact inverse.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch

from kvfold.config import JoltConfig
from kvfold.core.jl import Distribution
from kvfold.core.budget import (
    Cell,
    Bisect,
    Plan,
)
from kvfold.core.base import (
    Payload,
    Stats,
    Compressor,
)
from kvfold.core.residual import (
    Residual,
    decode_residual,
    encode_residual,
)
from kvfold.core.svd import Exact, Randomized
from kvfold.core.tucker import (
    Tucker,
    partial_tucker_st_hosvd,
    reconstruct_partial_tucker,
)

__all__ = ["Jolt"]

log = logging.getLogger(__name__)


_ALLOWED_DISTRIBUTIONS: tuple[Distribution, ...] = ("gaussian", "rademacher", "sparse")


def _coerce_distribution(value: object) -> Distribution:
    """Validate and narrow a distribution tag to the :data:`Distribution` literal.

    Accepts the in-memory form (``"gaussian"`` / ``"rademacher"`` /
    ``"sparse"``) and the legacy stringified form from older payloads.
    Raises :class:`ValueError` for unknown values.
    """
    if value in _ALLOWED_DISTRIBUTIONS:
        return value
    raise ValueError(f"unknown distribution tag: {value!r}")


@dataclass
class JoLTFactors:
    """Internal per-cell (key or value) JoLT factors.

    Attributes:
        tucker: partial Tucker decomposition result.
        residual: encoded JL-rotated residual.
        allocation: allocator decision for this cell.
    """

    tucker: Tucker
    residual: Residual | None
    allocation: Any


class Jolt(Compressor):
    """Paper-faithful JoLT compressor."""

    @classmethod
    def default_config(cls) -> JoltConfig:
        return JoltConfig()

    """Paper-faithful JoLT compressor.

    Args:
        ratio: target compression ratio (e.g. ``3.0``).
        bits: residual bit-widths the allocator can choose from.
        dtype: dtype of stored Tucker factors (``fp16`` or ``fp32``).
        jl_distribution: ``"gaussian"`` or ``"rademacher"``.
        allocator: optional pre-built :class:`Bisect`. If ``None``,
            one is constructed from ``ratio`` and ``bits``.
        svd: optional shared :class:`SVD` (for deterministic seeding).
        symmetric_quant: symmetric vs. asymmetric quantization.
        per_channel_quant: per-channel scales.
        group_size: per-group scale size (only if not per_channel).
        layer_groups: number of layer groups (kept at 1 for compatibility;
            the paper uses G=1).
    """

    method: str = "jolt"

    def __init__(
        self,
        *,
        ratio: float = 3.0,
        bits: tuple[int, ...] = (0, 2, 4, 8),
        dtype: torch.dtype = torch.float16,
        jl_distribution: Distribution = "gaussian",
        allocator: Bisect | None = None,
        decomposer: Decomposer | None = None,
        symmetric_quant: bool = True,
        per_channel_quant: bool = True,
        group_size: int | None = None,
        layer_groups: int = 1,
        seed: int = 0,
        **unused: Any,
    ) -> None:
        super().__init__()
        if ratio <= 1.0:
            raise ValueError(f"ratio must be > 1.0; got {ratio}")
        self.ratio = float(ratio)
        self.bits = tuple(bits)
        self.dtype = dtype
        self.jl_distribution = jl_distribution
        # Ponytail: allocator defaults to fp16 element size; pass
        # ``element_size_bytes=...`` to match a fp32 cache (allocator
        # bytes budget then doubles and the achieved ratio lands at the
        # user-requested target instead of half).
        self.allocator = allocator or Bisect(
            target_ratio=ratio,
            bits_grid=self.bits,
            factor_dtype_bytes=dtype.itemsize,
        )
        self.decomposer = decomposer or Exact()
        self.symmetric_quant = symmetric_quant
        self.per_channel_quant = per_channel_quant
        self.group_size = group_size
        self.layer_groups = int(layer_groups)
        self.seed = int(seed)
        self.stats_history: list[Stats] = []

    # ------------------------------------------------------------------
    # Compress / decompress
    # ------------------------------------------------------------------

    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[Payload, Payload]:
        """Allocator-driven JoLT compression of a (K, V) pair.

        Steps:
            1. Build two :class:`Cell` objects (one for K, one for V)
               describing shape and candidate bit-widths.
            2. Call :meth:`Bisect.optimize` to obtain the
               Lagrange-multiplier-optimal ``(rT, rd, bits)`` per cell.
            3. Run :meth:`compress_cell` for each side, producing a
               :class:`JoLTFactors` (Tucker factors + JL-rotated residual
               payload + the allocation decision).
            4. Wrap the factors via :meth:`build_payload` and accumulate
               per-call statistics.

        Args:
            key: K tensor of shape ``(m, T, dh)``.
            value: V tensor of the same shape.

        Returns:
            A ``(key_payload, value_payload)`` pair ready for storage in
            :class:`~kvfold.store.compress.CompressedKVCache`.

        Raises:
            ValueError: if K and V have different shapes or are not 3-D.
        """
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
        alloc_result: Plan = self.allocator.optimize(cells)

        k_alloc, v_alloc = alloc_result.allocations

        k_factors = self.compress_cell(key, k_alloc)
        v_factors = self.compress_cell(value, v_alloc)

        k_payload = self.build_payload(key, k_factors)
        v_payload = self.build_payload(value, v_factors)

        elapsed = (time.perf_counter() - t0) * 1000
        new_stats = Stats(
            compress_ms=elapsed,
            bytes_original=key.numel() * key.element_size() * 2,
            bytes_compressed=k_payload.bytes_compressed + v_payload.bytes_compressed,
            cell_count=len(self.stats_history) + 1,
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
        self.stats_history.append(new_stats)
        return k_payload, v_payload

    def restore(
        self,
        key_payload: Payload,
        value_payload: Payload,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of :meth:`compress`.

        Reconstructs the Tucker core, adds back the JL-rotated residual
        (if present) and casts to ``payload.dtype``. Reconstruction cost
        is dominated by a single ``einsum("mar,ta,dr->mtd", ...)``.

        Args:
            key_payload: payload produced by :meth:`compress`.
            value_payload: payload produced by :meth:`compress`.

        Returns:
            ``(K, V)`` reconstructed to original dtype and shape.
        """
        k = self.reconstruct_payload(key_payload)
        v = self.reconstruct_payload(value_payload)
        return k, v

    def stats(self) -> dict[str, Any]:
        """Return aggregated stats across all compress()/restore() calls.

        Stats are stored in a list rather than mutated on the instance, so
        the compressor is safe to share across threads.

        Notes:
            ``last_stats.decompress_time_ms`` is updated lazily: only the
            first decompress after each compress updates the field. This
            matches how the paper's Table 2 reports compress-only numbers.
        """
        latest = self.stats_history[-1].to_dict() if self.stats_history else {}
        return {
            "method": self.method,
            "history_size": len(self.stats_history),
            **latest,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def compress_cell(
        self,
        x: torch.Tensor,
        allocation: Any,
    ) -> JoLTFactors:
        """Compress a single K or V side per the allocator's decision.

        Steps:
            1. Run ST-HOSVD at ranks ``(r_token, r_feature)``.
            2. Reconstruct the partial Tucker approximation and compute
               the residual ``R = x - x̂``.
            3. JL-rotate and quantise ``R`` at ``bits`` via
               :func:`~kvfold.core.residual.encode_residual`.
               The ``bits == 0`` branch skips the residual entirely
               (pure-Tucker mode).
            4. Return a :class:`JoLTFactors` bundling Tucker factors,
               residual payload, and the allocation record.

        Args:
            x: the K or V tensor (``(m, T, dh)``).
            allocation: an :class:`Allocation` with ``r_token``,
                ``r_feature``, ``bits``.

        Returns:
            A :class:`JoLTFactors` triple (tucker, residual, allocation).
        """
        rt = int(allocation.r_token)
        rd = int(allocation.r_feature)
        b = int(allocation.bits)
        # ST-HOSVD
        tucker = partial_tucker_st_hosvd(
            x,
            r_token=rt,
            r_feature=rd,
            svd=self.decomposer,
        )
        # Residual
        recon = reconstruct_partial_tucker(tucker, x.shape)
        residual_tensor = (x - recon).contiguous()
        if b == 0:
            residual: Residual | None = encode_residual(
                residual_tensor,
                bits=0,
                seed=self.seed,
                distribution=self.jl_distribution,
            )
        else:
            residual = encode_residual(
                residual_tensor,
                bits=b,
                seed=self.seed,
                distribution=self.jl_distribution,
                symmetric=self.symmetric_quant,
                per_channel=self.per_channel_quant,
                group_size=self.group_size,
            )
        return JoLTFactors(tucker=tucker, residual=residual, allocation=allocation)

    def build_payload(self, original: torch.Tensor, factors: JoLTFactors) -> Payload:
        """Wrap :class:`JoLTFactors` into a serialisable payload.

        Casts the Tucker core and bases to ``dtype``, serialises
        the residual (if any) under the ``residual_*`` keys, and
        records the tail Frobenius mass for diagnostics.

        Args:
            original: the unmodified K/V tensor, kept only for ``dtype``
                and ``shape``.
            factors: output of :meth:`compress_cell`.

        Returns:
            A :class:`Payload` containing ``data`` (Tucker core
            + bases + residual buffers) and ``metadata`` (ranks, bit
            widths, residual config, tail mass).
        """
        # Cast factors to storage dtype.
        core = factors.tucker.core.to(self.dtype).contiguous()
        u_token = factors.tucker.u_token.to(self.dtype).contiguous()
        u_feature = factors.tucker.u_feature.to(self.dtype).contiguous()

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
            "core_dtype": "fp16" if self.dtype == torch.float16 else "fp32",
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

        return Payload(
            method="jolt",
            shape=tuple(original.shape),
            dtype=original.dtype,
            metadata=meta,
            data=data,
            stats=Stats(
                bytes_original=original.numel() * original.element_size(),
                bytes_compressed=core.numel() * core.element_size()
                + u_token.numel() * u_token.element_size()
                + u_feature.numel() * u_feature.element_size(),
            ),
        )

    def reconstruct_payload(self, payload: Payload) -> torch.Tensor:
        """Reconstruct a single K or V tensor from a JoLT payload.

        Steps:
            1. Promote the stored ``core``, ``u_token``, ``u_feature`` to
               fp32 and run :func:`reconstruct_partial_tucker`.
            2. If a residual was stored, decode it via
               :func:`decode_residual` and add it back to the Tucker
               reconstruction (the residual is in the original-dim
               space thanks to the JL projection's pseudoinverse).
            3. Cast back to ``payload.dtype``.

        Args:
            payload: a payload produced by :meth:`build_payload`.

        Returns:
            ``x`` reconstructed to ``payload.dtype``, shape ``payload.shape``.

        Raises:
            ValueError: if ``payload.method != "jolt"``.
        """
        if payload.method != "jolt":
            raise ValueError(f"JoLT cannot decode payload with method={payload.method!r}")
        core = payload.data["core"].to(torch.float32)
        u_token = payload.data["u_token"].to(torch.float32)
        u_feature = payload.data["u_feature"].to(torch.float32)
        x = reconstruct_partial_tucker(
            Tucker(
                core=core,
                u_token=u_token,
                u_feature=u_feature,
                token_sv=torch.empty(0),
                feature_sv=torch.empty(0),
                token_tail_mass=float(payload.metadata.get("tail_token_mass", 0.0)),
                feature_tail_mass=float(payload.metadata.get("tail_feature_mass", 0.0)),
            ),
            target_shape=tuple(int(d) for d in payload.shape),
        )

        # Add residual if present.
        if "residual_packed" in payload.data:
            res_dtype_int = int(payload.data["residual_dtype"].item())
            quant_dtype = f"int{res_dtype_int}" if res_dtype_int > 0 else "int0"
            original_shape = tuple(int(d) for d in payload.data["residual_original_shape"].tolist())
            original_last = int(payload.data["residual_original_last"].item())
            residual = Residual(
                projection_seed=int(payload.metadata["residual_seed"]),
                projection_distribution=_coerce_distribution(payload.metadata["residual_distribution"]),
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
