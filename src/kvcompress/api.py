"""kvcompress public API.

The high-level entrypoint is :func:`enable_compression`, which monkey-patches a
Hugging Face model so its KV cache is compressed transparently during
generation.

This module exposes:

* :class:`CompressionHandle` — handle returned by ``enable_compression`` used
  to disable compression, query stats, or swap methods at runtime.
* :func:`enable_compression` — entry point.
* :func:`disable_compression` — revert the patch.
* :func:`parse_target_memory` — helper that converts ``"25%"`` / ``0.25``
  to a ratio of ``4.0``.

The Hugging Face integration lives here rather than in ``adapters/`` because
this is the *only* function end-users need to know about.

Design notes:

* ``enable_compression`` is *not* idempotent: calling it twice on the same
  model raises a warning and returns the same handle. Use
  ``CompressionHandle.disable()`` first.
* ``enable_compression`` returns a :class:`CompressionHandle` rather than
  ``None`` because callers need to (a) disable compression later, (b)
  read cumulative stats. Returning ``None`` would force callers to use
  module-level state.
* The ``method`` string is the *only* required selection. All other knobs
  have sensible defaults that put the model in the paper's free zone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from kvcompress.adapters.huggingface import HuggingFaceAdapter

log = logging.getLogger(__name__)


MethodName = Literal[
    "jolt",
    "flashjolt",
    "lowrank",
    "int8",
    "int4",
    "int2",
    "fp8",
    "fp16",
    "identity",
]


@dataclass
class CompressionStats:
    """Aggregate statistics gathered across one compression session."""

    compress_calls: int = 0
    decompress_calls: int = 0
    bytes_original: int = 0
    bytes_compressed: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.bytes_compressed == 0:
            return 1.0
        return self.bytes_original / self.bytes_compressed

    @property
    def memory_saved_bytes(self) -> int:
        return max(0, self.bytes_original - self.bytes_compressed)


@dataclass
class CompressionHandle:
    """Handle returned by :func:`enable_compression`.

    Use it to disable the patch, query cumulative stats, or rebuild the
    compressor at runtime.
    """

    adapter: HuggingFaceAdapter
    model: Any
    stats: CompressionStats = field(default_factory=CompressionStats)

    def disable(self) -> None:
        """Disable compression and restore the original behaviour."""
        self.adapter.disable()
        log.info("kvcompress: disabled compression on %s", type(self.model).__name__)

    def stats_dict(self) -> dict[str, float]:
        return {
            "compress_calls": self.stats.compress_calls,
            "decompress_calls": self.stats.decompress_calls,
            "bytes_original": self.stats.bytes_original,
            "bytes_compressed": self.stats.bytes_compressed,
            "compression_ratio": self.stats.compression_ratio,
            "memory_saved_bytes": self.stats.memory_saved_bytes,
        }


def enable_compression(
    model: "PreTrainedModel",
    *,
    method: MethodName = "flashjolt",
    target_memory: str | float | None = None,
    compression_ratio: float | None = None,
    layer_groups: int = 1,
    bits: tuple[int, ...] = (0, 2, 4, 8),
    cache_implementation: str = "kvcompress",
    seed: int = 0,
    **kwargs: Any,
) -> CompressionHandle:
    """Enable transparent KV cache compression on a Hugging Face model.

    Args:
        model: a ``PreTrainedModel`` returned by ``AutoModelForCausalLM`` or
            similar.
        method: compressor name. One of ``jolt``, ``flashjolt``, ``lowrank``,
            ``int2``, ``int4``, ``int8``, ``fp8``, ``fp16``, ``identity``.
        target_memory: target memory as a fraction of original. Examples:
            ``"25%"`` (4× compression), ``"50%"`` (2×), or a float like
            ``0.25``. Mutually exclusive with ``compression_ratio``.
        compression_ratio: target compression ratio as a float (e.g. ``3.0``
            for 3×). Mutually exclusive with ``target_memory``.
        layer_groups: number of contiguous layer groups the allocator splits
            the model into. The paper uses ``G = 1`` by default; increase to
            give the allocator finer control.
        bits: tuple of allowed residual bit-widths the allocator can choose
            from. Default ``(0, 2, 4, 8)`` matches the paper.
        cache_implementation: name registered with HF's cache mechanism.
        seed: seed for randomized components.
        **kwargs: forwarded to the underlying compressor.

    Returns:
        :class:`CompressionHandle` used to disable compression or read stats.

    Raises:
        ValueError: if neither ``target_memory`` nor ``compression_ratio`` is
            provided, or if both are provided.
    """
    if (target_memory is None) == (compression_ratio is None):
        raise ValueError("Exactly one of `target_memory` or `compression_ratio` must be provided.")

    if target_memory is not None:
        compression_ratio = parse_target_memory(target_memory)
    elif compression_ratio is None:
        raise ValueError("unreachable")

    log.info(
        "kvcompress: enabling method=%s ratio=%.2fx on %s",
        method,
        compression_ratio,
        type(model).__name__,
    )

    from kvcompress.adapters.huggingface import HuggingFaceAdapter

    adapter = HuggingFaceAdapter(
        model=model,
        method=method,
        compression_ratio=compression_ratio,
        layer_groups=layer_groups,
        bits=bits,
        cache_implementation=cache_implementation,
        seed=seed,
        **kwargs,
    )
    handle = CompressionHandle(adapter=adapter, model=model)
    # Wire the stats object so the patched DynamicCache can update it.
    adapter.stats_ref = handle.stats  # type: ignore[assignment]
    adapter.enable()
    return handle


def disable_compression(handle: CompressionHandle) -> None:
    """Disable compression on a handle returned by :func:`enable_compression`."""
    handle.disable()


def parse_target_memory(value: str | float) -> float:
    """Convert a target-memory specification to a compression ratio.

    Examples:
        ``"25%"`` → ``4.0``
        ``"50%"`` → ``2.0``
        ``0.25`` → ``4.0``
    """
    if isinstance(value, (int, float)):
        if not 0 < value <= 1:
            raise ValueError(f"target_memory fraction must be in (0, 1], got {value}")
        return 1.0 / value
    s = str(value).strip()
    if s.endswith("%"):
        pct = float(s[:-1])
        if not 0 < pct <= 100:
            raise ValueError(f"target_memory percent must be in (0, 100], got {pct}")
        return 100.0 / pct
    raise ValueError(f"target_memory must be a fraction or 'N%' string, got {value!r}")
