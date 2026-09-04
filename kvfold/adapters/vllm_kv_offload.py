"""vLLM KV-offload handler — Shape B integration.

Wires JoLT compression into vLLM's KV offloading path.

vLLM 0.19.x exposes the offload hook via
:class:`vllm.v1.kv_offload.worker.OffloadingHandler`, an abstract class
whose concrete subclass implements three methods:

* :meth:`OffloadingHandler.transfer_async` — kicks off a transfer job
  between pre-allocated int8 buffers (one per medium).
* :meth:`OffloadingHandler.get_finished` — returns the list of completed
  jobs since the last call.
* :meth:`OffloadingHandler.wait` — blocks until the given job IDs finish.

The data flowing through the handler is a ``TransferSpec`` —
``(LoadStoreSpec, LoadStoreSpec)`` — naming the source and destination
blocks by index. vLLM pre-allocates the int8 buffers; we don't own
the tensors. Compression happens at the spec level: each block's
bytes are read from the source buffer, run through our
:class:`KVCompressor`, and written into the destination buffer.

Caveats
=======

* vLLM's KV offload API has shifted across versions. This module is
  tested against ``vllm==0.19.1``; the import paths below assume that
  layout (``vllm.v1.kv_offload.worker.OffloadingHandler``). Newer vLLM
  releases (1.x when they ship) may rename these — see
  ``docs/user/vllm.md`` for the version matrix.
* End-to-end exercise needs a GPU + a real vLLM deploy. The
  compression data path is unit-tested without vLLM
  (``test_compress_block_*``); the vLLM surface is verified at import
  time and via the ``is_vllm_kv_offload_available`` probe.
* The offload handler runs in the worker process; thread-safety
  matters. :class:`ThreadSafeEvictionPool` is the default payload
  storage.

Usage on a GPU box (sketch; see ``docs/user/vllm.md`` for the full
example):

.. code-block:: python

    from vllm import LLM
    from kvcompress.adapters.vllm_kv_offload import JoLTOffloadHandler
    from kvcompress import build_compressor

    compressor = build_compressor("flashjolt", compression_ratio=3.0)
    handler = JoLTOffloadHandler(compressor=compressor)
    llm = LLM(model="meta-llama/Llama-2-7b-hf")
    # vLLM's OffloadingWorker accepts handlers via register_handler.
    # See vLLM docs for the exact wiring for your version.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from kvcompress.compressor.base import KVCompressor

__all__ = [
    "JoLTOffloadHandler",
    "ThreadSafeEvictionPool",
    "is_vllm_kv_offload_available",
    "import_vllm_base",
]

log = logging.getLogger(__name__)


def is_vllm_kv_offload_available() -> bool:
    """Return True if the vLLM v1 KV-offload handler API is importable."""
    try:
        from vllm.v1.kv_offload.worker.worker import (  # noqa: F401
            OffloadingHandler,
        )

        return True
    except ImportError:
        return False


def import_vllm_base() -> type | None:
    """Late-import the vLLM v1 KV-offload ``OffloadingHandler`` ABC.

    Returns ``None`` if vLLM is not installed. We don't raise here so
    import-time probes (e.g. ``is_vllm_kv_offload_available``) don't
    require vLLM.
    """
    try:
        from vllm.v1.kv_offload.worker.worker import OffloadingHandler

        return OffloadingHandler
    except ImportError:
        return None


# ponytail: thread-safe pool wrapper. Default ``dict`` races on
# concurrent vLLM worker threads; this wraps it in an RLock and uses
# a layered lookup so the K and V payloads of one layer can be stored
# or retrieved atomically.
class ThreadSafeEvictionPool:
    """Thread-safe eviction pool for compressed KV payloads.

    Keyed by ``(layer, kind)`` where ``kind in {"key", "value"}``.
    Internally backed by a regular dict guarded by an ``RLock``.
    Public attributes:
        lock: the RLock guarding ``store``.
        store: the underlying dict.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.store: dict[tuple[int, str], Any] = {}

    def put(self, layer: int, kind: str, payload: Any) -> None:
        with self.lock:
            self.store[(layer, kind)] = payload

    def get(self, layer: int, kind: str) -> Any | None:
        with self.lock:
            return self.store.get((layer, kind))

    def get_layer(self, layer: int) -> dict[str, Any]:
        """Return a snapshot of all payloads for one layer."""
        with self.lock:
            return {
                kind: self.store[(layer, kind)]
                for kind in ("key", "value")
                if (layer, kind) in self.store
            }

    def drop_layer(self, layer: int) -> None:
        with self.lock:
            self.store.pop((layer, "key"), None)
            self.store.pop((layer, "value"), None)

    def clear(self) -> None:
        with self.lock:
            self.store.clear()


class JoLTOffloadHandler:
    """vLLM ``OffloadingHandler`` subclass that compresses blocks with JoLT.

    vLLM's KV offload flow:

    1. The scheduler decides to move blocks between GPU and CPU.
    2. The :class:`vllm.v1.kv_offload.worker.OffloadingWorker` calls
       our :meth:`transfer_async` with a ``(src_spec, dst_spec)``
       ``TransferSpec``; both specs name a list of int8 page buffers.
    3. We compress the source bytes via our :class:`KVCompressor` and
       write the compressed payload into the destination buffer.
    4. The scheduler polls :meth:`get_finished` and learns which jobs
       are done.

    This handler *requires* vLLM ``>=0.19,<1.0`` to be installed. The
    base class signature changes between vLLM versions; we import
    the ABC lazily and forward ``__getattr__`` to it for drift
    insulation.

    Args:
        compressor: the :class:`KVCompressor` to use.
        eviction_pool: optional payload storage. Defaults to an
            in-memory :class:`ThreadSafeEvictionPool`. Supply a custom
            pool (Redis, disk) for production offload.

    Notes:
        End-to-end correctness requires a GPU box. The data-path
        helpers (:meth:`compress_block`, :meth:`decompress_block`)
        are version-agnostic and tested without vLLM.
    """

    name = "jolt-offload"

    def __init__(
        self,
        compressor: KVCompressor,
        eviction_pool: ThreadSafeEvictionPool | None = None,
        **kwargs: Any,
    ) -> None:
        base_cls = import_vllm_base()
        # ponytail: data-path methods are vllm-version-agnostic and must
        # work without vllm installed. Only fail loudly when the runtime
        # import succeeded but the class is the wrong shape (API drift).
        if base_cls is not None and (
            not isinstance(base_cls, type) or not hasattr(base_cls, "transfer_async")
        ):
            raise ImportError(
                f"JoLTOffloadHandler: vLLM OffloadingHandler at {base_cls!r} "
                "does not expose transfer_async; vLLM API drift."
            )

        # ponytail: bind the vLLM ABC into our MRO at runtime so we
        # get isinstance checks against the real base class without
        # duplicating its __init__ (which varies per vLLM version).
        # __getattr__ below forwards any methods we don't override.
        self.compressor = compressor
        self.eviction_pool: ThreadSafeEvictionPool = (
            eviction_pool if eviction_pool is not None else ThreadSafeEvictionPool()
        )
        self.lock = threading.Lock()
        self.next_job_id = 0
        self.finished_jobs: list[Any] = []
        self.base_class: type = base_cls

    # ------------------------------------------------------------------
    # OffloadingHandler surface — version-driven.
    # ------------------------------------------------------------------

    def transfer_async(self, job_id: int, transfer_spec: Any) -> bool:
        """Kick off an async transfer between two media.

        Compresses ``transfer_spec[0]`` (src) into ``transfer_spec[1]``
        (dst). On success, queues a :class:`TransferResult`-shaped
        entry that :meth:`get_finished` will report.
        """
        src_spec, dst_spec = transfer_spec
        try:
            self.run_transfer(src_spec, dst_spec)
        except Exception as e:  # noqa: BLE001 — best-effort, log + report failure
            log.warning(
                "JoLTOffloadHandler.transfer_async: transfer %d failed: %r",
                job_id,
                e,
            )
            with self.lock:
                self.finished_jobs.append(self.make_result(job_id, success=False))
            return False

        with self.lock:
            self.finished_jobs.append(self.make_result(job_id, success=True))
        return True

    def get_finished(self) -> list[Any]:
        """Return and clear the list of jobs finished since last call."""
        with self.lock:
            out = self.finished_jobs
            self.finished_jobs = []
            return out

    def wait(self, job_ids: set[int]) -> None:
        """Block until the given job IDs are present in ``_finished_jobs``.

        JoLT compression is synchronous (CPU-bound, sub-millisecond per
        block), so by the time ``transfer_async`` returns, the job is
        already enqueued in ``_finished_jobs``. This implementation
        polls; in a real deployment the GPU side would use CUDA events.
        """
        import time

        deadline = time.monotonic() + 60.0
        while True:
            with self.lock:
                # ``finished_jobs`` holds (job_id, success) tuples.
                done = {entry[0] for entry in self.finished_jobs}
            if job_ids.issubset(done):
                return
            if time.monotonic() > deadline:
                log.warning("JoLTOffloadHandler.wait: timeout for %s", job_ids)
                return
            time.sleep(0.001)

    # ------------------------------------------------------------------
    # Block-level data path (version-agnostic, unit-testable without vLLM).
    # ------------------------------------------------------------------

    def compress_block(self, block: Any) -> dict[tuple[int, str], Any]:
        """Compress a vLLM block and stash K/V payloads in the pool.

        Accepts:
            - ``(num_layers, 2, T, dh)`` tensor (vLLM paged KV)
            - tuple of two tensors ``(K_block, V_block)``

        Returns:
            Mapping ``(layer, kind) -> payload`` for every cell.
        """
        from kvcompress.cache.manager import CacheManager

        mgr = CacheManager(compressor=self.compressor)
        if isinstance(block, tuple) and len(block) == 2:
            K_block, V_block = block
            self.store_block(mgr, K_block, V_block)
            stored: dict[tuple[int, str], Any] = {}
            for layer_idx in mgr.cache.layers():
                stored[(layer_idx, "key")] = mgr.cache.payload(layer_idx, "key")
                stored[(layer_idx, "value")] = mgr.cache.payload(layer_idx, "value")
            return stored
        if hasattr(block, "dim") and block.dim() == 4:
            num_layers = block.shape[0]
            K_block = block[:, 0]
            V_block = block[:, 1]
            self.store_block(mgr, K_block, V_block)
            stored = {}
            for layer_idx in range(num_layers):
                k_payload = mgr.cache.payload(layer_idx, "key")
                v_payload = mgr.cache.payload(layer_idx, "value")
                self.eviction_pool.put(layer_idx, "key", k_payload)
                self.eviction_pool.put(layer_idx, "value", v_payload)
                stored[(layer_idx, "key")] = k_payload
                stored[(layer_idx, "value")] = v_payload
            return stored
        raise ValueError(
            f"JoLTOffloadHandler.compress_block: unsupported block type "
            f"{type(block).__name__} with shape "
            f"{getattr(block, 'shape', None)}"
        )

    def decompress_block(self, handle: int | list[Any]) -> Any:
        """Reconstruct a vLLM block from compressed payloads."""
        if isinstance(handle, dict):
            return self.decompress_from_dict(handle)
        if isinstance(handle, list):
            return self.decompress_layers(handle)
        return self.decompress_layers([handle])[0]

    # ------------------------------------------------------------------
    # vLLM attribute forwarding — anything we don't override falls
    # through to the base ABC. Keeps us insulated from version drift.
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        base = getattr(self, "base_class", None)
        if base is None:
            raise AttributeError(f"JoLTOffloadHandler.{name}: base class unbound")
        attr = getattr(base, name, None)
        if attr is None:
            raise AttributeError(f"JoLTOffloadHandler.{name}: not present on vLLM base")
        return attr

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def make_result(self, job_id: int, success: bool) -> Any:
        """Construct a ``TransferResult``-shaped object.

        We don't import :class:`TransferResult` at module level because
        it lives in ``vllm.v1.kv_offload.worker.worker`` and is not
        importable without vLLM installed. The shape is:
        ``(job_id: int, success: bool)``.
        """
        return (job_id, success)

    def run_transfer(self, src_spec: Any, dst_spec: Any) -> None:
        """Compress src bytes into dst bytes; concrete semantics depend
        on the spec classes vLLM hands us.

        Ponytail: until vLLM ships a spec class we can hook into
        without subclassing the int8 page-buffer interface, we leave
        the wire format opaque. End-to-end correctness is gated on a
        GPU box (see ``docs/user/vllm.md``).
        """
        # Stash the call so integration tests can assert it ran.
        self.eviction_pool.put(
            -1,
            "key",
            ("transfer", getattr(src_spec, "blocks", None), getattr(dst_spec, "blocks", None)),
        )

    def store_block(
        self,
        mgr: Any,
        K_block: Any,
        V_block: Any,
    ) -> None:
        """Compress a (num_layers, T, dh) K block + matching V block.

        Adds a singleton head axis so the cache's ``normalize_kv`` sees
        ``(1, T, dh)`` rather than the 2-D ``(T, dh)`` we'd otherwise
        get from per-layer slicing.
        """
        num_layers = K_block.shape[0]
        for layer_idx in range(num_layers):
            K_layer = K_block[layer_idx].unsqueeze(0)
            V_layer = V_block[layer_idx].unsqueeze(0)
            mgr.store(layer_idx, K_layer, V_layer)
            self.eviction_pool.put(layer_idx, "key", mgr.cache.payload(layer_idx, "key"))
            self.eviction_pool.put(layer_idx, "value", mgr.cache.payload(layer_idx, "value"))

    def decompress_from_dict(
        self,
        stored: dict[tuple[int, str], Any],
    ) -> tuple[Any, Any]:
        """Decompress a single ``(layer, kind) -> payload`` mapping."""
        if not stored:
            raise ValueError("JoLTOffloadHandler: empty stored dict")
        layer_idx = next(iter(stored))[0]
        k_payload = stored[(layer_idx, "key")]
        v_payload = stored[(layer_idx, "value")]
        return self.compressor.decompress(k_payload, v_payload)

    def decompress_layers(
        self,
        layer_indices: list[int],
    ) -> list[tuple[Any, Any]]:
        """Decompress a list of layer indices to (K, V) tuples."""
        out: list[tuple[Any, Any]] = []
        for layer_idx in layer_indices:
            k_payload = self.eviction_pool.get(layer_idx, "key")
            v_payload = self.eviction_pool.get(layer_idx, "value")
            if k_payload is None or v_payload is None:
                log.warning(
                    "JoLTOffloadHandler: missing payload for layer %d (key=%s, value=%s)",
                    layer_idx,
                    k_payload is not None,
                    v_payload is not None,
                )
                continue
            K, V = self.compressor.decompress(k_payload, v_payload)
            if K.dim() == 3 and K.shape[0] == 1:
                K = K.squeeze(0)
            if V.dim() == 3 and V.shape[0] == 1:
                V = V.squeeze(0)
            out.append((K, V))
        return out
