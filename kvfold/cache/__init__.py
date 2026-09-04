"""Compressed KV cache storage layer.

This subpackage owns the *physical* side of KV cache compression: how the
:class:`~kvcompress.compressor.base.CompressedPayload` objects produced by a
compressor are stored, indexed, evicted, and reported on.

Three objects cooperate:

* :class:`~kvcompress.cache.compress.CompressedKVCache` — the low-level
  layer-indexed store. Holds the actual tensors and exposes
  :meth:`~CompressedKVCache.store` / :meth:`~CompressedKVCache.retrieve`.
* :class:`~kvcompress.cache.manager.CacheManager` — a thin facade that adds
  bookkeeping (which layers are live, what the manager has seen) on top
  of the layer-indexed cache. This is the object the HF adapter holds.
* :class:`~kvcompress.cache.metadata.CompressionMetadata` and
  :class:`~kvcompress.cache.metadata.LayerCompression` — serializable
  dataclasses that record the per-layer allocation decisions so a cache
  can be saved to safetensors, transmitted, or reloaded across processes.

The cache is **not** model-aware: it does not know about attention heads,
RoPE, or sliding windows. Those concerns live in the compressor and the
adapter. The cache is purely a key-value store indexed by layer number.
"""

from kvcompress.cache.compress import CompressedKVCache
from kvcompress.cache.manager import CacheManager
from kvcompress.cache.metadata import CompressionMetadata, LayerCompression

__all__ = [
    "CacheManager",
    "CompressedKVCache",
    "CompressionMetadata",
    "LayerCompression",
]
