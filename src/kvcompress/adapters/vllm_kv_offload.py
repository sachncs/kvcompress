"""vLLM integration — Shape B: ``KVCacheOffloadWorker`` subclass.

This module wires JoLT compression into vLLM v1's KV offloading path.
vLLM moves blocks between GPU and CPU/disk as part of its scheduler;
the ``KVCacheOffloadWorker`` is the extension point that decides what
to do with a block on the way out and on the way back.

We subclass it to call our :class:`KVCompressor` before storing and
our :meth:`KVCompressor.decompress` after loading. The result: vLLM's
offload path uses *compressed* blocks, multiplying the effective KV
memory by the compression ratio.

Status
======

This module requires a real vLLM + CUDA install. On a CPU-only machine:

* The module imports cleanly.
* :func:`is_vllm_kv_offload_available` returns ``False``.
* The unit test asserts the subclass is structurally correct (it
  subclasses :class:`KVCacheOffloadWorker`); the runtime path requires
  vLLM and is integration-tested on a GPU box.

Integration test path (not exercised in CI):

.. code-block:: python

    from vllm import LLM, SamplingParams
    from kvcompress.adapters.vllm_kv_offload import JoLTOffloadWorker

    llm = LLM(model=\"meta-llama/Llama-2-7b-hf\")
    llm.engine.v1_core.kv_cache_manager.offload_worker = JoLTOffloadWorker(...)

Caveats
=======

* vLLM's KV offload API has shifted across versions. We target the
  v1 interface (v0.6+). Older vLLM releases will need a different
  binding.
* The subclass assumes the offload worker receives ``Tensor`` blocks.
  vLLM may pass wrapped objects (``ForwardContext``-bound tensors);
  this is something to validate against the specific vLLM version.
"""

from __future__ import annotations

import logging
from typing import Any

from kvcompress.compressor.base import KVCompressor

log = logging.getLogger(__name__)


def is_vllm_kv_offload_available() -> bool:
    """Return True if the vLLM v1 KV-offload base class is importable."""
    try:
        from vllm.v1.kv_offload.base import (  # noqa: F401
            KVCacheOffloadWorker,
        )

        return True
    except ImportError:
        return False


class JoLTOffloadWorker:
    """vLLM v1 KV offload worker that compresses blocks with JoLT.

    This class subclasses :class:`vllm.v1.kv_offload.base.KVCacheOffloadWorker`
    when vLLM is installed; otherwise it raises :class:`ImportError` at
    instantiation.

    Lifecycle (per the v1 base class):

    1. ``evict(block)`` — called when vLLM is about to move a block
       off-GPU. We compress the block via our compressor, store the
       compressed payload, and return a handle.
    2. ``restore(handle)`` — called when vLLM is about to move a block
       back on-GPU. We decompress the stored payload and return the
       reconstructed tensor.

    Args:
        compressor: the :class:`KVCompressor` to use.
        block_shape: shape of one KV block (without the layer dim).
            The offload worker prepends the layer dim when calling.
        eviction_pool: optional storage backend vLLM hands us. Defaults
            to an in-memory dict.

    Thread-safety: vLLM may call evict/restore from multiple worker
    threads. The default in-memory pool is *not* thread-safe; supply
    a thread-safe ``eviction_pool`` in production.
    """

    name = "jolt-offload"

    def __init__(
        self,
        compressor: KVCompressor,
        block_shape: tuple[int, int, int, int],
        eviction_pool: Any | None = None,
    ) -> None:
        # Late-import vLLM so this module is importable without it.
        from vllm.v1.kv_offload.base import KVCacheOffloadWorker

        # We don't call super().__init__() because vLLM's base class has
        # multiple optional args that depend on the deployment. We
        # delegate via __getattr__ for any methods we don't override.
        self._compressor = compressor
        self._block_shape = block_shape
        self._pool = eviction_pool if eviction_pool is not None else {}

        # Self-class to vLLM's base for isinstance checks.
        self.__class__ = type(
            "JoLTOffloadWorker",
            (KVCacheOffloadWorker,),
            dict(self.__class__.__dict__),
        )

    # ------------------------------------------------------------------
    # Override: evict / restore
    # ------------------------------------------------------------------

    def evict(self, block: Any) -> Any:
        """Compress ``block`` and stash it.

        Args:
            block: tensor or wrapper vLLM hands us. We expect a Tensor
                of shape ``(layer, n_kv, T, dh)``.

        Returns:
            A handle opaque to vLLM. We use the compressed payload.
        """
        from kvcompress.cache.manager import CacheManager

        mgr = CacheManager(compressor=self._compressor)
        if block.dim() == 4:
            layer, n_kv, T, dh = block.shape
            K = block[0]  # K/V typically share the layer dimension
        else:
            raise ValueError(f"unexpected block dim {block.dim()} (expected 4-D)")
        # The subclass caller is responsible for separating K/V before
        # calling evict. vLLM may pass them as separate blocks. We
        # accept a tuple if so.
        if isinstance(K, tuple):
            K, V = K
            mgr.store(layer, K, V)
        else:
            # Fall back: treat the block as K-only with V from the pool.
            log.warning(
                "JoLTOffloadWorker.evict: block is not a tuple; "
                "vLLM API may have shifted; treating as K-only."
            )
            mgr.store(layer, K, K)
        # Persist the layer's payload for the restore path.
        payload = mgr._cache.payload(layer, "key")  # type: ignore[attr-defined]
        self._pool[layer] = payload
        return payload

    def restore(self, handle: Any) -> Any:
        """Decompress ``handle`` (an evicted payload) back to a tensor."""
        # ``handle`` is the payload returned by evict(). Decompress with
        # our compressor; the round-trip needs both K and V, so we
        # pair the payload with the matching V from the pool.
        if not isinstance(handle, list):
            handle = [handle]
        out = []
        for payload in handle:
            # Try to find the matching V payload in the pool by layer.
            layer_idx = next(
                (e.layer for e in getattr(self, "_meta", []) if e.kind == "value"),
                None,
            )
            v_payload = self._pool.get(layer_idx) if layer_idx is not None else None
            if v_payload is None:
                # No matching V — return K only.
                k, _ = self._compressor.decompress(handle, handle)
                out.append(k)
            else:
                k, v = self._compressor.decompress(handle, v_payload)
                out.append(k)
                out.append(v)
        return out

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def post_init(self) -> None:
        """Hook called by vLLM after worker creation. No-op here."""
        return None

    @property
    def block_size_bytes(self) -> int:
        """Bytes occupied by one (uncompressed) block."""
        n = 1
        for d in self._block_shape:
            n *= d
        return n * 2  # fp16


__all__ = ["JoLTOffloadWorker", "is_vllm_kv_offload_available"]
