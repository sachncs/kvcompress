"""Runtime memory pool — reusable tensor buffers.

A simple pool that hands out contiguous tensors of a given shape/dtype and
recycles them across calls. Reduces allocator pressure during repeated
compression/decompression cycles.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch

log = logging.getLogger(__name__)


class MemoryPool:
    """Pool of reusable torch tensors keyed by (shape, dtype, device).

    Args:
        max_per_key: maximum number of tensors to keep per key. Older
            tensors are dropped when the cap is hit.
    """

    def __init__(self, max_per_key: int = 32) -> None:
        self.max_per_key = int(max_per_key)
        self._pool: dict[tuple[tuple[int, ...], torch.dtype, torch.device], list[torch.Tensor]] = (
            defaultdict(list)
        )

    def acquire(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Return a tensor of ``shape`` from the pool (or allocate new)."""
        key = (tuple(shape), dtype, torch.device(device))
        bucket = self._pool.get(key)
        if bucket:
            t = bucket.pop()
            t.zero_()
            return t
        return torch.empty(shape, dtype=dtype, device=device)

    def release(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool."""
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)
        bucket = self._pool[key]
        if len(bucket) >= self.max_per_key:
            return  # drop
        bucket.append(tensor)

    def clear(self) -> None:
        self._pool.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "num_keys": len(self._pool),
            "total_buffers": sum(len(b) for b in self._pool.values()),
        }


__all__ = ["MemoryPool"]