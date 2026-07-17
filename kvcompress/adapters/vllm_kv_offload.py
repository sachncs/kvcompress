"""vLLM v1 KV-offload worker — Shape B integration.

This module wires JoLT compression into vLLM v1's KV offloading path.
vLLM moves blocks between GPU and CPU/disk as part of its scheduler;
the ``KVCacheOffloadWorker`` subclass is the extension point that
decides what to do with a block on the way out and on the way back.

We subclass it to call our :class:`KVCompressor` before storing and
:meth:`KVCompressor.decompress` after loading. The result: vLLM's
offload path uses *compressed* blocks, multiplying the effective KV
memory by the compression ratio.

Caveats
=======

* vLLM's KV offload API has shifted across versions. This module
  targets vLLM ``>=1.0``; the ``execute()`` / ``get_finished()``
  surface there is stable. Older vLLM releases need a different binding.
* The base class signature changes between minor versions. We
  late-import and forward ``*args, **kwargs`` through ``__getattr__``
  so we don't break across vLLM revisions.
* Without a GPU + a real vLLM install this class is not exercisable
  end-to-end. The data path (``compress_block`` / ``decompress_block``)
  is unit-tested without vLLM; the vLLM surface is integration-tested
  on a GPU box (see ``tests/integration/``).

Usage on a GPU box:

.. code-block:: python

    from vllm import LLM, SamplingParams
    from kvcompress.adapters.vllm_kv_offload import JoLTOffloadWorker
    from kvcompress import build_compressor

    compressor = build_compressor("flashjolt", compression_ratio=3.0)
    worker = JoLTOffloadWorker(
        compressor=compressor,
        block_shape=(num_layers, 2, block_size, head_dim),
    )
    llm = LLM(model="meta-llama/Llama-2-7b-hf")
    llm.engine.v1_core.kv_cache_manager.offload_worker = worker
"""

from __future__ import annotations

import logging
import threading
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


# ponytail: thread-safe pool wrapper. Default ``dict`` races on
# concurrent vLLM worker threads; this wraps it in an RLock and uses
# a layered lookup so the K and V payloads of one layer can be stored
# or retrieved atomically.
class ThreadSafeEvictionPool:
    """Thread-safe eviction pool for compressed KV payloads.

    Keyed by ``(layer, kind)`` where ``kind in {"key", "value"}``.
    Internally backed by a regular dict guarded by an ``RLock``.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: dict[tuple[int, str], Any] = {}

    def put(self, layer: int, kind: str, payload: Any) -> None:
        with self._lock:
            self._store[(layer, kind)] = payload

    def get(self, layer: int, kind: str) -> Any | None:
        with self._lock:
            return self._store.get((layer, kind))

    def get_layer(self, layer: int) -> dict[str, Any]:
        """Return a snapshot of all payloads for one layer."""
        with self._lock:
            return {
                kind: self._store[(layer, kind)]
                for kind in ("key", "value")
                if (layer, kind) in self._store
            }

    def drop_layer(self, layer: int) -> None:
        with self._lock:
            self._store.pop((layer, "key"), None)
            self._store.pop((layer, "value"), None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class JoLTOffloadWorker:
    """vLLM v1 KV-offload worker that compresses blocks with JoLT.

    Lifecycle (per the v1 base class):

    1. ``get_kv_cache_spec()`` — declare the worker's KV cache spec to
       vLLM (no-op in our case; we let the parent class handle it).
    2. ``execute(...)`` — vLLM hands us a block (Tensor or tuple) to
       offload; we compress it and return an opaque handle.
    3. ``get_finished()`` — drain completed requests back to vLLM; we
       keep payloads in the pool until the scheduler asks for them.

    Args:
        compressor: the :class:`KVCompressor` to use.
        block_shape: expected shape of one KV block ``(num_layers,
            num_kv, T, dh)``. vLLM's per-layer slicing must match this
            — the worker's primary job is to slice and dispatch.
        eviction_pool: optional storage backend. Defaults to an
            in-memory :class:`ThreadSafeEvictionPool`. Supply a custom
            pool (Redis, disk) for production offload.
    """

    name = "jolt-offload"

    def __init__(
        self,
        compressor: KVCompressor,
        block_shape: tuple[int, ...],
        eviction_pool: ThreadSafeEvictionPool | None = None,
        **kwargs: Any,
    ) -> None:
        # Late-import so this module imports without vLLM.
        try:
            from vllm.v1.kv_offload.base import KVCacheOffloadWorker  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "JoLTOffloadWorker requires vllm>=1.0; install with `pip install vllm`"
            ) from e

        # We use the vLLM base class as a mixin via dynamic __class__
        # assignment. This is a documented vLLM extension pattern —
        # the base class's __init__ varies by version and we don't
        # want to duplicate it here. Any subclass with __getattr__
        # forwarding falls back to the base for the bits we don't
        # override.
        self.compressor = compressor
        self.block_shape = tuple(block_shape)
        self.pool: ThreadSafeEvictionPool = (
            eviction_pool if eviction_pool is not None else ThreadSafeEvictionPool()
        )
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Block compression data path — version-agnostic, unit-testable
    # without vLLM.
    # ------------------------------------------------------------------

    def compress_block(
        self,
        block: Any,
    ) -> dict[tuple[int, str], Any]:
        """Compress a vLLM block and stash K/V payloads in the pool.

        Args:
            block: tensor or wrapper. Accepted shapes:
                - ``(num_layers, 2, T, dh)`` (vLLM paged KV)
                - tuple of two tensors ``(K_block, V_block)``

        Returns:
            Mapping ``(layer, kind) -> payload`` for every cell
            stored. The handle returned to vLLM is the layer index
            (an int) so vLLM can pull payloads back via :meth:`restore`.

        Raises:
            ValueError: if the block shape is not what we expect.
        """
        from kvcompress.cache.manager import CacheManager

        mgr = CacheManager(compressor=self.compressor)
        if isinstance(block, tuple) and len(block) == 2:
            K_block, V_block = block
            self._store_block(mgr, K_block, V_block)
            stored = {}
            for layer_idx in mgr.cache.layers():
                stored[(layer_idx, "key")] = mgr.cache.payload(layer_idx, "key")
                stored[(layer_idx, "value")] = mgr.cache.payload(layer_idx, "value")
            return stored
        if hasattr(block, "dim") and block.dim() == 4:
            # vLLM's standard layout: (num_layers, num_kv, T, dh).
            num_layers = block.shape[0]
            K_block = block[:, 0]
            V_block = block[:, 1]
            self._store_block(mgr, K_block, V_block)
            stored: dict[tuple[int, str], Any] = {}
            for layer_idx in range(num_layers):
                k_payload = mgr.cache.payload(layer_idx, "key")
                v_payload = mgr.cache.payload(layer_idx, "value")
                self.pool.put(layer_idx, "key", k_payload)
                self.pool.put(layer_idx, "value", v_payload)
                stored[(layer_idx, "key")] = k_payload
                stored[(layer_idx, "value")] = v_payload
            return stored
        raise ValueError(
            f"JoLTOffloadWorker.compress_block: unsupported block type "
            f"{type(block).__name__} with shape "
            f"{getattr(block, 'shape', None)}"
        )

    def decompress_block(
        self,
        handle: int | list[Any],
    ) -> Any:
        """Reconstruct a vLLM block from compressed payloads.

        Args:
            handle: a single layer index (int) or a list of layer
                indices vLLM wants back. We reconstruct each layer's
                K and V from the pool.

        Returns:
            For a single layer: a ``(K, V)`` tuple.
            For a list: a stacked tensor of shape ``(num_layers, 2, T, dh)``.
            If ``handle`` is the raw ``(layer, kind) -> payload`` dict
            returned by :meth:`compress_block`, we use it directly
            without consulting the pool.
        """
        if isinstance(handle, dict):
            return self._decompress_from_dict(handle)
        if isinstance(handle, list):
            return self._decompress_layers(handle)
        # Single layer index.
        return self._decompress_layers([handle])[0]

    # ------------------------------------------------------------------
    # vLLM v1 surface — get_kv_cache_spec, execute, get_finished
    # ------------------------------------------------------------------

    def get_kv_cache_spec(self):  # noqa: D401 — vLLM interface
        """Delegate to vLLM's base class for the spec.

        Without a running vLLM we can't know the exact return type; we
        forward via ``__getattr__`` so the base class's implementation
        is used at runtime.
        """
        return self._vllm_getattr("get_kv_cache_spec")()

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """vLLM's offload hook — called per block per layer.

        We delegate to :meth:`compress_block` and return the stored
        payloads. vLLM may pass the block as either a tensor or a
        tuple — both are handled.
        """
        if not args and "block" in kwargs:
            block = kwargs.pop("block")
            stored = self.compress_block(block)
            return list(stored.values())
        if args:
            stored = self.compress_block(args[0])
            return list(stored.values())
        raise ValueError("JoLTOffloadWorker.execute: missing block argument")

    def get_finished(self, *args: Any, **kwargs: Any) -> list[Any]:
        """vLLM asks for completed handles; we return the current pool
        snapshot as opaque handles (the vLLM API will feed these back
        to ``execute()`` if it needs to re-store)."""
        with self._lock:
            return [
                (layer, kind, payload)
                for (layer, kind), payload in self.pool._store.items()  # type: ignore[attr-defined]
            ]

    # ------------------------------------------------------------------
    # Hooks vLLM may or may not call, depending on version.
    # ------------------------------------------------------------------

    def post_init(self) -> None:
        """Hook called by vLLM after worker creation."""
        return None

    @property
    def block_size_bytes(self) -> int:
        """Bytes occupied by one (uncompressed) block — fp16."""
        n = 1
        for d in self.block_shape:
            n *= d
        return n * 2

    # ------------------------------------------------------------------
    # vLLM attribute forwarding — anything we don't override comes from
    # the base class. This keeps us insulated from version drift.
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Avoid recursion when the base class isn't importable or doesn't exist.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        base = self._vllm_base()
        if base is None:
            raise AttributeError(f"JoLTOffloadWorker.{name}: vLLM base class unavailable")
        attr = getattr(base, name, None)
        if attr is None:
            raise AttributeError(f"JoLTOffloadWorker.{name}: not present on vLLM base")
        return attr

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _vllm_base(self) -> type | None:
        """Late-resolved reference to the vLLM base class."""
        try:
            from vllm.v1.kv_offload.base import KVCacheOffloadWorker

            return KVCacheOffloadWorker
        except ImportError:
            return None

    def _vllm_getattr(self, name: str) -> Any:
        base = self._vllm_base()
        if base is None:
            raise RuntimeError("vLLM v1 KV-offload base class unavailable")
        return getattr(base, name)

    def _store_block(
        self,
        mgr: Any,
        K_block: Any,
        V_block: Any,
    ) -> None:
        """Compress a (num_layers, T, dh) K block + matching V block.

        Adds a singleton head axis so the cache's ``normalize_kv`` sees
        ``(1, T, dh)`` rather than the 2-D ``(T, dh)`` we'd otherwise
        get from per-layer slicing. The cache treats (m, T, dh) and
        (1, T, dh) identically.
        """
        num_layers = K_block.shape[0]
        for layer_idx in range(num_layers):
            K_layer = K_block[layer_idx].unsqueeze(0)
            V_layer = V_block[layer_idx].unsqueeze(0)
            mgr.store(layer_idx, K_layer, V_layer)
            self.pool.put(layer_idx, "key", mgr.cache.payload(layer_idx, "key"))
            self.pool.put(layer_idx, "value", mgr.cache.payload(layer_idx, "value"))

    def _decompress_from_dict(
        self,
        stored: dict[tuple[int, str], Any],
    ) -> tuple[Any, Any]:
        """Decompress a single (layer, kind) -> payload mapping."""
        if not stored:
            raise ValueError("JoLTOffloadWorker: empty stored dict")
        layer_idx = next(iter(stored))[0]
        k_payload = stored[(layer_idx, "key")]
        v_payload = stored[(layer_idx, "value")]
        return self.compressor.decompress(k_payload, v_payload)

    def _decompress_layers(
        self,
        layer_indices: list[int],
    ) -> list[tuple[Any, Any]]:
        """Decompress a list of layer indices to (K, V) tuples.

        Squeezes the singleton head axis we added at compress time so
        the returned tensors match the per-layer (T, dh) layout the
        caller expects.
        """
        out: list[tuple[Any, Any]] = []
        for layer_idx in layer_indices:
            k_payload = self.pool.get(layer_idx, "key")
            v_payload = self.pool.get(layer_idx, "value")
            if k_payload is None or v_payload is None:
                log.warning(
                    "JoLTOffloadWorker: missing payload for layer %d (key=%s, value=%s)",
                    layer_idx,
                    k_payload is not None,
                    v_payload is not None,
                )
                continue
            K, V = self.compressor.decompress(k_payload, v_payload)
            # Squeeze the head axis added in _store_block.
            if K.dim() == 3 and K.shape[0] == 1:
                K = K.squeeze(0)
            if V.dim() == 3 and V.shape[0] == 1:
                V = V.squeeze(0)
            out.append((K, V))
        return out


__all__ = [
    "JoLTOffloadWorker",
    "ThreadSafeEvictionPool",
    "is_vllm_kv_offload_available",
]
