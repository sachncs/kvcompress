"""Protocols for pluggable compressed-cache storage backends.

The default implementation is the in-memory :class:`~kvfold.store.compress.Cache`.
Framework adapters depend on this narrow port rather than on a concrete cache,
which makes bounded, persistent, remote, and test-double backends explicit
extension points.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import torch

from kvfold.store.metadata import Meta


class CacheBackend(Protocol):
    """Storage contract required by :class:`~kvfold.store.manager.Pool`."""

    def store(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        group_id: int = 0,
        group_size: int = 1,
        bits: tuple[int, ...] = (0, 2, 4, 8),
        seed: int = 0,
    ) -> None:
        """Compress and store one layer's key/value tensors."""

    def retrieve(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct one layer's key/value tensors."""

    def clear(self) -> None:
        """Remove all stored layers."""

    def evict_layer(self, layer: int) -> None:
        """Remove one layer."""

    def memory_used(self) -> int:
        """Return logical compressed payload bytes."""

    def memory_original(self) -> int:
        """Return original uncompressed bytes."""

    def memory_resident(self) -> int:
        """Return resident tensor-storage bytes."""

    def compression_ratio(self) -> float:
        """Return original bytes divided by compressed bytes."""

    def stats(self) -> dict[str, Any]:
        """Return aggregate backend metrics."""

    def metadata(self) -> Meta:
        """Return the backend metadata snapshot."""

    def layers(self) -> Iterator[int]:
        """Iterate over stored layer IDs."""

    def __len__(self) -> int:
        """Return the number of stored layers."""

    def __contains__(self, layer: int) -> bool:
        """Return whether a layer is present."""


__all__ = ["CacheBackend"]
