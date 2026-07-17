"""Tests for the Shape B vLLM KV-offload handler.

The data-path methods (:meth:`compress_block`, :meth:`decompress_block`)
are version-agnostic and tested here without vLLM by injecting a stub
base class. The vLLM surface (``is_vllm_kv_offload_available`` and the
runtime import path) is also exercised end-to-end.
"""

from __future__ import annotations

import sys
import threading
import types
from typing import Any

import pytest
import torch

from kvcompress import IdentityCompressor, JoLTCompressor
from kvcompress.adapters.vllm_kv_offload import (
    JoLTOffloadHandler,
    ThreadSafeEvictionPool,
    is_vllm_kv_offload_available,
)


def test_is_vllm_kv_offload_available_returns_bool() -> None:
    assert isinstance(is_vllm_kv_offload_available(), bool)


def test_module_imports_without_vllm() -> None:
    """The module must import on systems without vLLM."""
    import kvcompress.adapters.vllm_kv_offload  # noqa: F401


def test_handler_attributes() -> None:
    """The handler exposes the expected public attributes."""
    comp = JoLTCompressor(compression_ratio=3.0)
    handler = JoLTOffloadHandler(compressor=comp, block_shape=(4, 2, 16, 8))
    assert handler.name == "jolt-offload"
    assert hasattr(handler, "compress_block")
    assert hasattr(handler, "decompress_block")
    assert hasattr(handler, "transfer_async")
    assert hasattr(handler, "get_finished")
    assert hasattr(handler, "wait")
    # Default pool is thread-safe.
    assert isinstance(handler.eviction_pool, ThreadSafeEvictionPool)


def test_handler_instantiation_with_real_vllm() -> None:
    """When vLLM is installed, the handler imports the real base ABC
    and binds it into the MRO via __getattr__ forwarding.
    """
    pytest.importorskip("vllm")
    comp = JoLTCompressor(compression_ratio=3.0)
    handler = JoLTOffloadHandler(compressor=comp)
    assert handler.name == "jolt-offload"
    assert handler._base_class is not None  # type: ignore[attr-defined]


def test_handler_instantiation_raises_without_vllm() -> None:
    """On systems without vLLM, instantiation should fail cleanly."""
    import builtins

    from kvcompress import JoLTCompressor

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("vllm") or "vllm" in name:
            raise ImportError(f"simulated missing module: {name}")
        return real_import(name, *args, **kwargs)

    comp = JoLTCompressor(compression_ratio=3.0)
    builtins.__import__ = fake_import
    try:
        with pytest.raises(ImportError):
            JoLTOffloadHandler(compressor=comp)
    finally:
        builtins.__import__ = real_import


def test_thread_safe_pool_round_trip() -> None:
    """Concurrent put/get across multiple threads must not corrupt state."""
    pool = ThreadSafeEvictionPool()
    n_threads = 16
    n_layers = 32

    barrier = threading.Barrier(n_threads)

    def worker(thread_id: int) -> None:
        barrier.wait()
        for layer in range(n_layers):
            pool.put(layer, "key", (thread_id, layer))
            pool.put(layer, "value", (thread_id, layer + 1000))
            assert pool.get(layer, "key") is not None
            assert pool.get(layer, "value") is not None

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Every layer has both K and V set.
    for layer in range(n_layers):
        snap = pool.get_layer(layer)
        assert "key" in snap
        assert "value" in snap


def test_compress_decompress_round_trip_separates_kv() -> None:
    """Regression: compress_block must not store K as both K and V."""
    comp = IdentityCompressor(factor_dtype=torch.float32)
    handler = JoLTOffloadHandler(compressor=comp)
    # vLLM's standard layout: (num_layers, num_kv, T, dh).
    torch.manual_seed(0)
    num_layers, _num_kv, T, dh = 2, 2, 4, 8
    K_orig = torch.randn(num_layers, T, dh)
    V_orig = torch.randn(num_layers, T, dh)
    block = torch.stack([K_orig, V_orig], dim=1)  # (num_layers, 2, T, dh)
    stored = handler.compress_block(block)
    assert len(stored) == 2 * num_layers
    for layer in range(num_layers):
        assert (layer, "key") in stored
        assert (layer, "value") in stored
        # K and V payloads must differ (identity keeps them distinct).
        assert not torch.equal(
            stored[(layer, "key")].data["value"],
            stored[(layer, "value")].data["value"],
        )
    # Restore each layer; K and V come back distinct.
    for layer in range(num_layers):
        K, V = handler.decompress_block(layer)
        assert torch.allclose(K, K_orig[layer])
        assert torch.allclose(V, V_orig[layer])
        # K != V across the originals.
        assert not torch.equal(K, V)


def test_compress_block_handles_tuple_input() -> None:
    """Accepts ``(K_block, V_block)`` tuple as well as the 4-D tensor."""
    comp = IdentityCompressor(factor_dtype=torch.float32)
    handler = JoLTOffloadHandler(compressor=comp)
    torch.manual_seed(0)
    K_orig = torch.randn(2, 4, 8)
    V_orig = torch.randn(2, 4, 8)
    stored = handler.compress_block((K_orig, V_orig))
    assert len(stored) == 4
    for layer in range(2):
        K, V = handler.decompress_block(layer)
        assert torch.allclose(K, K_orig[layer])
        assert torch.allclose(V, V_orig[layer])


def test_compress_block_rejects_unknown_layout() -> None:
    """A 3-D block (no layer axis) must raise loudly, not silently store."""
    comp = IdentityCompressor()
    handler = JoLTOffloadHandler(compressor=comp)
    bad = torch.randn(2, 4, 8)  # no layer axis
    with pytest.raises(ValueError, match="unsupported block type"):
        handler.compress_block(bad)


def test_decompress_block_returns_list_of_tuples_for_list_input() -> None:
    """vLLM may pass a list of layer indices; we return one (K, V) per layer."""
    comp = IdentityCompressor(factor_dtype=torch.float32)
    handler = JoLTOffloadHandler(compressor=comp)
    torch.manual_seed(0)
    K_orig = torch.randn(3, 4, 8)
    V_orig = torch.randn(3, 4, 8)
    block = torch.stack([K_orig, V_orig], dim=1)
    handler.compress_block(block)
    out = handler.decompress_block([0, 2])
    assert len(out) == 2
    K0, V0 = out[0]
    K2, V2 = out[1]
    assert torch.allclose(K0, K_orig[0])
    assert torch.allclose(V0, V_orig[0])
    assert torch.allclose(K2, K_orig[2])
    assert torch.allclose(V2, V_orig[2])


def test_transfer_async_records_job_completion() -> None:
    """A successful transfer_async enqueues a finished job; get_finished drains."""
    comp = IdentityCompressor(factor_dtype=torch.float32)
    handler = JoLTOffloadHandler(compressor=comp)

    class _FakeSpec:
        def __init__(self, blocks: list[int]) -> None:
            self.blocks = blocks

    src = _FakeSpec([0, 1])
    dst = _FakeSpec([0, 1])
    ok = handler.transfer_async(7, (src, dst))
    assert ok is True
    finished = handler.get_finished()
    assert len(finished) == 1
    job_id, success = finished[0]
    assert job_id == 7
    assert success is True
    # get_finished drains.
    assert handler.get_finished() == []


def test_transfer_async_records_failure_on_exception() -> None:
    """A transfer_async that raises still enqueues a (job_id, success=False)."""
    from kvcompress.adapters import vllm_kv_offload as mod

    comp = IdentityCompressor(factor_dtype=torch.float32)
    handler = JoLTOffloadHandler(compressor=comp)

    # Force _run_transfer to raise.
    def boom(self, src, dst):  # noqa: ARG001
        raise RuntimeError("simulated failure")

    original = mod.JoLTOffloadHandler._run_transfer
    mod.JoLTOffloadHandler._run_transfer = boom
    try:
        ok = handler.transfer_async(99, (None, None))
        assert ok is False
        finished = handler.get_finished()
        assert len(finished) == 1
        assert finished[0] == (99, False)
    finally:
        mod.JoLTOffloadHandler._run_transfer = original


def test_wait_returns_when_jobs_finish() -> None:
    """``wait`` polls and returns once all requested job IDs are in
    the finished list."""
    comp = IdentityCompressor(factor_dtype=torch.float32)
    handler = JoLTOffloadHandler(compressor=comp)

    class _FakeSpec:
        blocks: list[int] = []

    handler._finished_jobs.append((1, True))  # type: ignore[attr-defined]
    handler._finished_jobs.append((2, True))  # type: ignore[attr-defined]
    handler.wait({1, 2})  # returns immediately


def test_wait_times_out_on_missing_jobs() -> None:
    """``wait`` returns on timeout instead of looping forever."""
    comp = IdentityCompressor(factor_dtype=torch.float32)
    handler = JoLTOffloadHandler(compressor=comp)
    # The default 60s deadline is too long for tests; monkey-patch.
    import time as _time

    real_sleep = _time.sleep
    _time.sleep = lambda s: None  # noqa: ARG005
    real_mono = _time.monotonic

    counter = {"n": 0}

    def fast_mono() -> float:
        counter["n"] += 1
        # Return increasing time so the deadline triggers quickly.
        return counter["n"] * 1000

    _time.monotonic = fast_mono
    try:
        # Empty finished_jobs — wait must return on timeout, not hang.
        handler.wait({999})
    finally:
        _time.sleep = real_sleep
        _time.monotonic = real_mono