"""Base classes and protocols for KV cache compressors.

A :class:`KVCompressor` takes raw key/value tensors at one layer and produces a
:class:`CompressedPayload` that captures the layout, ranks, residual bit-widths,
and serialized factors. Decompression is a deterministic function of the
payload. Every concrete compressor in :mod:`kvcompress.compressor` is a
subclass.

The contract is intentionally minimal:

* ``compress(K, V) -> (kp, vp)`` returns two payloads.
* ``decompress(kp, vp) -> (K', V')`` returns reconstructions.
* :meth:`estimate_size` returns the bytes occupied by one payload.

Compressors don't own caches — that's :class:`CompressedKVCache`'s job.
This separation lets us swap compression algorithms without touching the
cache, and vice versa.

Thread-safety: compressors are stateless across calls (all per-call
state lives in ``CompressedPayload``), so a single instance is safe to
share across threads. The seeded RNG inside SVD/JL is a fresh
``torch.Generator`` per call.
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

log = logging.getLogger(__name__)


@dataclass
class CompressorStats:
    """Per-call statistics about a compressor invocation.

    Attributes:
        compress_time_ms: wall time of the compress() call.
        decompress_time_ms: wall time of the most recent decompress().
        bytes_original: size of the input tensor in bytes (element count ×
            element size).
        bytes_compressed: total bytes occupied by the payload.
        reconstruction_error: relative Frobenius error of the most recent
            reconstruction, ``||X - X̂||_F / ||X||_F``. ``None`` until a
            reconstruct happens.
    """

    compress_time_ms: float = 0.0
    decompress_time_ms: float = 0.0
    bytes_original: int = 0
    bytes_compressed: int = 0
    reconstruction_error: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compress_time_ms": self.compress_time_ms,
            "decompress_time_ms": self.decompress_time_ms,
            "bytes_original": self.bytes_original,
            "bytes_compressed": self.bytes_compressed,
            "compression_ratio": self.compression_ratio,
            "reconstruction_error": self.reconstruction_error,
            **self.extra,
        }

    @property
    def compression_ratio(self) -> float:
        if self.bytes_compressed == 0:
            return 1.0
        return self.bytes_original / self.bytes_compressed


@dataclass
class CompressedPayload:
    """Generic container for any compressed representation.

    Every concrete compressor stores its representation inside ``data`` with
    a class-specific schema; ``metadata`` is the serializable side of the
    payload (ranks, bit-widths, layout) that survives the round trip through
    safetensors.

    Attributes:
        method: identifier of the compressor that produced this payload
            (e.g. ``"jolt"``, ``"flashjolt"``, ``"lowrank"``).
        shape: shape of the original tensor.
        dtype: original tensor dtype.
        metadata: compressor-specific serializable metadata.
        data: dict mapping names to tensors or python scalars (e.g. factors,
            packed residuals, scales).
        stats: per-call statistics captured at compression time.
    """

    method: str
    shape: tuple[int, ...] | torch.Size
    dtype: torch.dtype
    metadata: dict[str, Any]
    data: dict[str, Any]
    stats: CompressorStats = field(default_factory=CompressorStats)

    @property
    def bytes_compressed(self) -> int:
        """Total bytes occupied by all tensor entries in ``data``.

        Python scalars in ``metadata`` contribute zero bytes; they are
        negligible compared to tensor factors.
        """
        total = 0
        for v in self.data.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, torch.Tensor):
                        total += vv.numel() * vv.element_size()
        return total

    @property
    def bytes_original(self) -> int:
        """Bytes of the original (uncompressed) tensor.

        Computed as ``prod(shape) * dtype_bytes``. Use ``CompressionStats.bytes_original``
        when you want the *cumulated* original bytes across many calls.
        """
        n = 1
        for d in self.shape:
            n *= d
        return n * torch.tensor([], dtype=self.dtype).element_size()


class KVCompressor(abc.ABC):
    """Abstract base for all KV cache compressors.

    Concrete subclasses implement :meth:`compress` and :meth:`decompress`.
    The two are required to be (approximately) inverses:

    .. code-block:: python

        payload = c.compress(k, v)
        k_hat, v_hat = c.decompress(payload)
        assert torch.allclose(k, k_hat, atol=...)  # within numerical noise
    """

    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abc.abstractmethod
    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[CompressedPayload, CompressedPayload]:
        """Compress a (key, value) pair into two :class:`CompressedPayload`.

        Args:
            key: tensor of shape ``(m, T, dh)`` where ``m`` merges head and
                layer, ``T`` is the token axis, ``dh`` is per-head feature
                dim.
            value: same shape as ``key``.

        Returns:
            Two payloads (key payload, value payload).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decompress(
        self,
        key_payload: CompressedPayload,
        value_payload: CompressedPayload,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of :meth:`compress`."""
        raise NotImplementedError

    def estimate_size(self, payload: CompressedPayload) -> int:
        """Return bytes occupied by ``payload``."""
        return payload.bytes_compressed

    def stats(self) -> dict[str, Any]:
        """Return aggregated stats for all calls so far on this instance."""
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# A factory callable for pluggable compressor construction.
CompressorFactory = Callable[..., KVCompressor]
