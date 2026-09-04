"""Compressed KV cache storage layer.

This subpackage owns the *physical* side of KV cache compression: how the
:class:`~kvfold.core.base.Payload` objects produced by a
compressor are stored, indexed, evicted, and reported on.

Three objects cooperate:

* :class:`~kvfold.store.compress.Cache` — the low-level
  layer-indexed store. Holds the actual tensors and exposes
  :meth:`~Cache.store` / :meth:`~Cache.retrieve`.
* :class:`~kvfold.store.manager.Pool` — a thin facade that adds
  bookkeeping (which layers are live, what the manager has seen) on top
  of the layer-indexed cache. This is the object the HF adapter holds.
* :class:`~kvfold.store.metadata.Meta` and
  :class:`~kvfold.store.metadata.LayerMeta` — serializable
  dataclasses that record the per-layer allocation decisions so a cache
  can be saved to safetensors, transmitted, or reloaded across processes.

The cache is **not** model-aware: it does not know about attention heads,
RoPE, or sliding windows. Those concerns live in the compressor and the
adapter. The cache is purely a key-value store indexed by layer number.
"""

from kvfold.store.compress import Cache
from kvfold.store.manager import Pool
from kvfold.store.metadata import Meta, LayerMeta

__all__ = [
    "Pool",
    "Cache",
    "Meta",
    "LayerMeta",
]
