"""High-level cache orchestration.

The :class:`CacheManager` is the entry point used by the HF adapter. It owns
a :class:`CompressedKVCache` and a per-call record of which layers are live.

Why a manager on top of :class:`CompressedKVCache`:

* It tracks *which layers are live* — needed by the adapter's
  ``__getitem__`` path to know whether to reconstruct or pass through.
* It forwards :meth:`memory_used`, :meth:`compression_ratio`, and
  :meth:`stats` so the adapter can report unified metrics.
* It's the natural place to add cross-layer optimisations later (e.g.
  shared JL projections across groups) without changing the lower cache.

The manager is *not* model-aware; it does not know which tensor is K vs V
— both are passed through the underlying cache as opaque payloads.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

from kvcompress.cache.compress import CompressedKVCache
from kvcompress.cache.metadata import CompressionMetadata
from kvcompress.compressor.base import KVCompressor

log = logging.getLogger(__name__)


@dataclass
class CacheManager:
    """High-level cache facade.

    Args:
        compressor: compressor used to (de)compress.
        max_layers: optional LRU eviction cap.
        device: device to materialize reconstructed tensors on.
    """

    compressor: KVCompressor
    max_layers: int | None = None
    device: torch.device | str | None = None
    _cache: CompressedKVCache = field(init=False)
    _live_layers: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._cache = CompressedKVCache(
            compressor=self.compressor,
            max_layers=self.max_layers,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        self._cache.store(layer, key, value, **kwargs)
        if layer not in self._live_layers:
            self._live_layers.append(layer)

    def retrieve(
        self,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._cache.retrieve(layer)

    def clear(self) -> None:
        self._cache.clear()
        self._live_layers.clear()

    def evict(self, layer: int) -> None:
        self._cache.evict_layer(layer)
        self._live_layers = [entry for entry in self._live_layers if entry != layer]

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def memory_used(self) -> int:
        return self._cache.memory_used()

    def memory_original(self) -> int:
        return self._cache.memory_original()

    def compression_ratio(self) -> float:
        return self._cache.compression_ratio()

    def stats(self) -> dict[str, Any]:
        s = self._cache.stats()
        s["live_layers"] = list(self._live_layers)
        return s

    def metadata(self) -> CompressionMetadata:
        return self._cache.metadata()

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def layers(self) -> Iterator[int]:
        return self._cache.layers()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, layer: int) -> bool:
        return layer in self._cache


__all__ = ["CacheManager"]
