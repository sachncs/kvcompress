"""High-level cache orchestration.

The :class:`Pool` is the entry point used by the HF adapter. It owns
a :class:`Cache` and a per-call record of which layers are live.

Why a manager on top of :class:`Cache`:

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
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

from kvfold.core.base import Compressor
from kvfold.store.compress import Cache
from kvfold.store.metadata import Meta
from kvfold.store.ports import CacheBackend

log = logging.getLogger(__name__)


@dataclass
class Pool:
    """High-level cache facade.

    Args:
        compressor: compressor used to (de)compress.
        max_layers: optional LRU eviction cap.
        device: device to materialize reconstructed tensors on.
    """

    compressor: Compressor
    max_layers: int | None = None
    device: torch.device | str | None = None
    backend: CacheBackend | None = None
    cache: CacheBackend = field(init=False)
    live_layers: list[int] = field(default_factory=list, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the underlying :class:`Cache` from ``compressor``."""
        if self.backend is not None:
            self.cache = self.backend
        else:
            self.cache = Cache(
                compressor=self.compressor, max_layers=self.max_layers, device=self.device
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
        """Compress and store the K/V pair for one layer.

        Forwards to :meth:`Cache.store`. After a successful
        store, ``layer`` is added to the ``live_layers`` list (used by
        the HF adapter's ``__getitem__`` path).
        """
        with self._lock:
            self.cache.store(layer, key, value, **kwargs)
            # Cache.enforce_eviction() may remove an older layer, so derive this
            # view from the cache rather than appending indefinitely.
            self.live_layers = list(self.cache.layers())

    def retrieve(
        self,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct and return the K/V pair for ``layer``."""
        with self._lock:
            return self.cache.retrieve(layer)

    def clear(self) -> None:
        """Drop every entry and reset the ``live_layers`` list."""
        with self._lock:
            self.cache.clear()
            self.live_layers.clear()

    def evict(self, layer: int) -> None:
        """Remove ``layer`` from the cache and the ``live_layers`` list."""
        with self._lock:
            self.cache.evict_layer(layer)
            self.live_layers = list(self.cache.layers())

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def memory_used(self) -> int:
        """Bytes occupied by all stored payloads."""
        with self._lock:
            return self.cache.memory_used()

    def memory_original(self) -> int:
        """Sum of uncompressed bytes across all stored payloads."""
        with self._lock:
            return self.cache.memory_original()

    def memory_resident(self) -> int:
        """Backing tensor-storage bytes currently resident in the cache."""
        with self._lock:
            return self.cache.memory_resident()

    def compression_ratio(self) -> float:
        """Achieved ratio (original / compressed) across all stored payloads."""
        with self._lock:
            return self.cache.compression_ratio()

    def stats(self) -> dict[str, Any]:
        """Aggregate stats including the live-layer list."""
        with self._lock:
            s = self.cache.stats()
            s["live_layers"] = list(self.live_layers)
            return s

    def metadata(self) -> Meta:
        """Return the live :class:`Meta`."""
        with self._lock:
            return self.cache.metadata()

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def layers(self) -> Iterator[int]:
        """Iterate over live layer indices in insertion order."""
        with self._lock:
            return iter(list(self.cache.layers()))

    def __len__(self) -> int:
        """Number of layers currently stored."""
        with self._lock:
            return len(self.cache)

    def __contains__(self, layer: int) -> bool:
        """``layer in manager`` is ``True`` when the layer has an entry."""
        with self._lock:
            return layer in self.cache
