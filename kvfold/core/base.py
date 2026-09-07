"""Base classes for KV cache compressors.

A :class:`Compressor` takes raw key/value tensors at one layer and
produces a :class:`Payload` that captures the layout, ranks, residual
bit-widths, and serialized factors. Decompression is a deterministic
function of the payload. Every concrete compressor in :mod:`kvfold.core`
is a subclass.

The contract is intentionally minimal:

* ``compress(K, V) -> (kp, vp)`` returns two payloads.
* ``restore(kp, vp) -> (K', V')`` returns reconstructions.
* :meth:`estimate_size` returns the bytes occupied by one payload.

Compressors don't own caches — that's :class:`Cache`'s job. This
separation lets us swap compression algorithms without touching the
cache, and vice versa.

Thread-safety: compressors are stateless across calls (all per-call
state lives in ``Payload``), so a single instance is safe to share
across threads. The seeded RNG inside SVD/JL is a fresh
``torch.Generator`` per call.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from kvfold.errors import CacheValidationError, DTypeError, ShapeError

log = logging.getLogger(__name__)


DTYPE_BYTES: dict[torch.dtype, int] = {
    torch.float8_e4m3fn: 1,
    torch.float8_e5m2: 1,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.uint8: 1,
}
"""Bytes per element for every supported dtype."""


@dataclass
class Stats:
    """Per-call statistics about a compressor invocation.

    Attributes:
        compress_ms: wall time of the compress() call.
        restore_ms: wall time of the most recent restore().
        bytes_original: size of the input tensor in bytes (element count ×
            element size).
        bytes_compressed: total bytes occupied by the payload.
        reconstruction_error: relative Frobenius error of the most recent
            reconstruction, ``||X - X̂||F / ||X||F``. ``None`` until a
            reconstruct happens.
        cell_count: number of (K, V) cells processed so far.
        error_bound: documented worst-case error bound for this method
            (e.g. ``0.30`` for 2-bit residual). ``None`` when not set.
        extra: arbitrary method-specific counters.
    """

    compress_ms: float = 0.0
    restore_ms: float = 0.0
    bytes_original: int = 0
    bytes_compressed: int = 0
    reconstruction_error: float | None = None
    cell_count: int = 0
    error_bound: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise this stats record (flat dict for JSON / log lines)."""
        return {
            "compress_ms": self.compress_ms,
            "restore_ms": self.restore_ms,
            "bytes_original": self.bytes_original,
            "bytes_compressed": self.bytes_compressed,
            "compression_ratio": self.compression_ratio,
            "reconstruction_error": self.reconstruction_error,
            "cell_count": self.cell_count,
            "error_bound": self.error_bound,
            **self.extra,
        }

    @property
    def compression_ratio(self) -> float:
        """Achieved compression ratio ``B_original / B_compressed``.

        Returns ``1.0`` if the compressed size is zero; never raises on
        divide-by-zero.
        """
        if self.bytes_compressed == 0:
            return 1.0
        return self.bytes_original / self.bytes_compressed


@dataclass
class Payload:
    """Generic container for any compressed representation.

    Every concrete compressor stores its representation inside ``data``
    with a class-specific schema; ``metadata`` is the serializable side
    of the payload (ranks, bit-widths, layout) that survives the round
    trip through safetensors.

    Attributes:
        method: identifier of the compressor that produced this payload.
        shape: shape of the original tensor.
        dtype: original tensor dtype.
        metadata: compressor-specific serializable metadata.
        data: dict mapping names to tensors or python scalars (e.g.
            factors, packed residuals, scales).
        stats: per-call statistics captured at compression time.
    """

    method: str
    shape: tuple[int, ...] | torch.Size
    dtype: torch.dtype
    metadata: dict[str, Any]
    data: dict[str, Any]
    stats: Stats = field(default_factory=Stats)

    @property
    def bytes_compressed(self) -> int:
        """Total bytes occupied by every tensor in ``data``.

        Walks the dict recursively so nested-dict values and list-of-tensor
        values are counted exactly once.
        """
        total = 0

        def walk(v: object) -> None:
            nonlocal total
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
            elif isinstance(v, dict):
                for vv in v.values():
                    walk(vv)
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    walk(vv)

        for v in self.data.values():
            walk(v)
        return total

    @property
    def bytes_original(self) -> int:
        """Bytes of the original (uncompressed) tensor."""
        n = 1
        for d in self.shape:
            n *= int(d)
        return n * DTYPE_BYTES.get(self.dtype, self.dtype.itemsize if hasattr(self.dtype, "itemsize") else 0)


class Compressor(abc.ABC):
    """Abstract base for all KV cache compressors.

    Concrete subclasses implement :meth:`compress` and :meth:`restore`.
    The two are required to be (approximately) inverses:

    .. code-block:: python

        payload = c.compress(k, v)
        k_hat, v_hat = c.restore(payload)
        assert torch.allclose(k, k_hat, atol=...)  # within numerical noise

    Subclasses register themselves with :class:`CompressorRegistry` via
    :meth:`__init_subclass__`.
    """

    method: str = "base"

    @property
    def name(self) -> str:
        """Backward-compat alias for :attr:`method`.

        Some pre-0.2.0 sites reference ``compressor.name``; the canonical
        name is :attr:`method`. This property keeps both readable.
        """
        return self.method

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @classmethod
    @abc.abstractmethod
    def default_config(cls) -> Any:
        """Return the default :class:`MethodConfig` for this compressor."""
        raise NotImplementedError

    @classmethod
    def validate(cls, key: torch.Tensor, value: torch.Tensor) -> None:
        """Validate that ``key`` and ``value`` satisfy the compressor's input contract.

        Default checks: same shape; both 3-D. Subclasses override to add
        dtype / device / range checks.
        """
        if key.shape != value.shape:
            raise CacheValidationError(layer=-1, expected=tuple(key.shape), actual=tuple(value.shape))
        if key.dim() != 3:
            raise ShapeError(
                f"expected 3-D tensor (m, T, dh); got {key.dim()}-D with shape {tuple(key.shape)}",
                hint="KV tensors must be (heads, length, head_dim)",
            )

    @abc.abstractmethod
    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[Payload, Payload]:
        """Compress a (key, value) pair into two :class:`Payload`.

        Args:
            key: tensor of shape ``(m, T, dh)``.
            value: same shape as ``key``.

        Returns:
            Two payloads (key payload, value payload).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def restore(
        self,
        key_payload: Payload,
        value_payload: Payload,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of :meth:`compress`."""
        raise NotImplementedError

    def estimate_size(self, payload: Payload) -> int:
        """Return bytes occupied by ``payload``."""
        return payload.bytes_compressed

    def clear_history(self) -> None:
        """Drop any per-instance stats history. Default: no-op."""
        return None

    def history(self) -> list[Stats]:
        """Return a list of per-call stats records. Default: empty."""
        return []

    def __repr__(self) -> str:
        return f"{type(self).__name__}(method={self.method!r})"


__all__ = ["Compressor", "Payload", "Stats", "DTYPE_BYTES"]
