"""kvfold public API.

The high-level entrypoint is :func:`enable_compression`, which monkey-patches a
Hugging Face model so its KV cache is compressed transparently during
generation.

This module exposes:

* :class:`CompressionHandle` — handle returned by ``enable_compression`` used
  to disable compression, query stats, or swap methods at runtime.
* :func:`enable_compression` — entry point.
* :func:`disable_compression` — revert the patch.
* :func:`build_compressor` — direct access to the compressor factory.
* :func:`supported_methods` — tuple of every supported method name.
* :func:`method_descriptors` — immutable registry bindings for tooling.
* :func:`parse_target_memory` — helper that converts a fraction such as
  ``0.25`` to a ratio of ``4.0``.

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
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import kvfold.core.builtins  # noqa: F401 — populates the compressor registry

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from kvfold.adapter.huggingface import HF

log = logging.getLogger(__name__)


MethodName = Literal[
    "jolt",
    "flash",
    "low",
    "int8",
    "int4",
    "int2",
    "fp8",
    "fp16",
    "bf16",
    "pass",
]


@dataclass
class CompressionStats:
    """Aggregate statistics gathered across one compression session."""

    compress_calls: int = 0
    decompress_calls: int = 0
    bytes_original: int = 0
    bytes_compressed: int = 0
    bytes_resident: int = 0
    _lock: Any = field(default_factory=threading.RLock, init=False, repr=False, compare=False)

    def record_compress(
        self,
        *,
        bytes_original: int,
        bytes_compressed: int,
        bytes_resident: int,
    ) -> None:
        """Atomically record one compression and its current cache totals."""
        with self._lock:
            self.compress_calls += 1
            self.bytes_original = bytes_original
            self.bytes_compressed = bytes_compressed
            self.bytes_resident = bytes_resident

    def record_decompress(self) -> None:
        """Atomically record one reconstruction."""
        with self._lock:
            self.decompress_calls += 1

    def snapshot(self) -> dict[str, int | float]:
        """Return an atomic immutable-value snapshot for metrics consumers."""
        with self._lock:
            return {
                "compress_calls": self.compress_calls,
                "decompress_calls": self.decompress_calls,
                "bytes_original": self.bytes_original,
                "bytes_compressed": self.bytes_compressed,
                "bytes_resident": self.bytes_resident,
                "ratio": self.compression_ratio,
                "memory_saved_bytes": self.memory_saved_bytes,
            }

    @property
    def compression_ratio(self) -> float:
        """Achieved compression ratio across the session (``B_orig / B_comp``).

        Returns ``1.0`` when no compression has happened yet, so the
        metric is always defined.
        """
        with self._lock:
            if self.bytes_compressed == 0:
                return 1.0
            return self.bytes_original / self.bytes_compressed

    @property
    def memory_saved_bytes(self) -> int:
        """Bytes saved versus the raw K/V cache. Floored at 0 (never negative)."""
        with self._lock:
            return max(0, self.bytes_original - self.bytes_compressed)


@dataclass(frozen=True)
class CompressionHandle:
    """Immutable handle returned by :func:`enable_compression`.

    Use it to disable the patch or query cumulative stats. The adapter owns
    mutable runtime state; callers receive snapshots rather than replacing
    the handle's references.
    """

    adapter: HF
    model: Any
    stats: CompressionStats = field(default_factory=CompressionStats)

    @property
    def is_active(self) -> bool:
        """``True`` if compression is still active on the model."""
        return bool(self.adapter.enabled)

    def disable(self) -> None:
        """Disable compression and restore the original behaviour."""
        self.adapter.disable()
        log.info("kvfold: disabled compression on %s", type(self.model).__name__)

    def stats_dict(self) -> dict[str, float]:
        """Return cumulative stats as a flat dict for logging / benchmarking.

        Includes both raw counters (call counts, byte totals) and the
        derived ``compression_ratio`` and ``memory_saved_bytes``.
        """
        return self.stats.snapshot()


def enable_compression(
    model: PreTrainedModel,
    *,
    method: MethodName = "flash",
    target_memory: float | None = None,
    ratio: float | None = None,
    layer_groups: int = 1,
    bits: tuple[int, ...] = (0, 2, 4, 8),
    cache_implementation: str = "dynamic",
    seed: int = 0,
    **kwargs: Any,
) -> CompressionHandle:
    """Enable transparent KV cache compression on a Hugging Face model.

    Args:
        model: a ``PreTrainedModel`` returned by ``AutoModelForCausalLM`` or
            similar.
        method: compressor name. One of ``jolt``, ``flash``, ``low``,
            ``int2``, ``int4``, ``int8``, ``fp8``, ``fp16``, ``bf16``, ``pass``.
        target_memory: target memory as a float fraction of original, such as
            ``0.25`` for 4× compression. Mutually exclusive with ``ratio``.
        ratio: target compression ratio as a float (e.g. ``3.0`` for 3×).
            Mutually exclusive with ``target_memory``.
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
        ValueError: if neither ``target_memory`` nor ``ratio`` is provided,
            or if both are provided.
    """
    if (target_memory is None) == (ratio is None):
        raise ValueError("Exactly one of `target_memory` or `ratio` must be provided.")

    if target_memory is not None:
        ratio = parse_target_memory(target_memory)
    elif ratio is None:
        raise ValueError("unreachable")

    # ``target_memory=1.0`` is identity — short-circuit so we don't pay
    # the allocator cost or hand the user a 3x default they didn't ask for.
    if ratio == 1.0:
        method = "pass"

    log.info(
        "kvfold: enabling method=%s ratio=%.2fx on %s",
        method,
        ratio,
        type(model).__name__,
    )

    from kvfold.adapter.huggingface import HF

    # Translate the public ``target_memory=1.0`` shortcut into the
    # passthrough compressor to avoid spinning up the allocator at ratio=1.
    extra: dict[str, Any] = dict(kwargs)
    if method == "pass":
        # Pass doesn't accept bits / seed / layer_groups — drop them
        # so build_compressor() doesn't reject them as unknown kwargs.
        for k in ("bits", "layer_groups"):
            extra.pop(k, None)

    adapter = HF(
        model=model,
        method=method,
        ratio=ratio,
        layer_groups=layer_groups,
        bits=bits,
        cache_implementation=cache_implementation,
        seed=seed,
        **extra,
    )
    handle = CompressionHandle(adapter=adapter, model=model)
    # Wire the stats object so the patched DynamicCache can update it.
    adapter.stats_ref = handle.stats
    adapter.enable()
    return handle


def disable_compression(handle: CompressionHandle) -> None:
    """Disable compression on a handle returned by :func:`enable_compression`."""
    handle.disable()


def build_compressor(method: str, **kwargs: Any) -> Any:
    """Construct a configured compressor from its public method name.

    Thin façade over :class:`kvfold.core.dispatch.CompressorRegistry` so
    users do not have to remember the internal registry path.

    Raises:
        UnsupportedMethodError: if ``method`` is not registered.
        MethodConfigError: if any kwarg is unknown or out of range.
    """
    from kvfold.core.dispatch import REGISTRY

    return REGISTRY.build(method, **kwargs)


def supported_methods() -> tuple[str, ...]:
    """Tuple of every compression method the dispatcher accepts."""
    from kvfold.core.dispatch import REGISTRY

    return REGISTRY.names()


def method_descriptors() -> tuple[Any, ...]:
    """Return immutable dispatcher descriptors for every public method.

    Each descriptor exposes the canonical method name, configuration class,
    implementation class, and capability set. Consumers that need a stable
    serialization format should project these objects to their own schema
    instead of inspecting registry internals.
    """
    from kvfold.core.dispatch import REGISTRY

    return tuple(REGISTRY.descriptor(name) for name in REGISTRY.names())


def parse_target_memory(value: float) -> float:
    """Convert a target-memory specification to a compression ratio.

    Example:
        ``0.25`` → ``4.0``
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"target_memory must be a numeric fraction, got {value!r}")
    if not 0 < value <= 1:
        raise ValueError(f"target_memory fraction must be in (0, 1], got {value}")
    return 1.0 / value
