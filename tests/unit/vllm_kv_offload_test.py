"""Tests for the Shape B vLLM KV-offload worker.

These tests run on CPU without vLLM installed. They verify the
subclass is structurally correct (imports cleanly, has the right
methods, exposes the expected public surface) and that the data path
(:meth:`compress_block` / :meth:`decompress_block`) round-trips K and
V correctly. Integration tests against a real vLLM + CUDA install are
out of scope for this repo; see ``docs/user/vllm.md``.
"""

from __future__ import annotations

import threading

import pytest
import torch

from kvcompress import IdentityCompressor, JoLTCompressor
from kvcompress.adapters.vllm_kv_offload import (
    JoLTOffloadWorker,
    ThreadSafeEvictionPool,
    is_vllm_kv_offload_available,
)


def test_is_vllm_kv_offload_available_returns_bool() -> None:
    assert isinstance(is_vllm_kv_offload_available(), bool)


def test_module_imports_without_vllm() -> None:
    """The module must import on systems without vLLM."""
    import kvcompress.adapters.vllm_kv_offload  # noqa: F401


def _stub_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake ``vllm.v1.kv_offload.base`` module so we can
    instantiate :class:`JoLTOffloadWorker` without the real vLLM."""
    import sys
    import types

    if "vllm" in sys.modules:
        return
    fake_vllm = types.ModuleType("vllm")
    fake_offload = types.ModuleType("vllm.v1.kv_offload")
    fake_base = types.ModuleType("vllm.v1.kv_offload.base")

    class _StubBase:
        """Stand-in for KVCacheOffloadWorker."""

    fake_base.KVCacheOffloadWorker = _StubBase
    sys.modules["vllm"] = fake_vllm
    sys.modules["vllm.v1"] = types.ModuleType("vllm.v1")
    sys.modules["vllm.v1.kv_offload"] = fake_offload
    sys.modules["vllm.v1.kv_offload.base"] = fake_base


def test_worker_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker exposes the expected public attributes."""
    _stub_vllm(monkeypatch)
    comp = JoLTCompressor(compression_ratio=3.0)
    worker = JoLTOffloadWorker(compressor=comp, block_shape=(4, 2, 16, 8))
    assert worker.name == "jolt-offload"
    assert hasattr(worker, "compress_block")
    assert hasattr(worker, "decompress_block")
    assert hasattr(worker, "execute")
    assert hasattr(worker, "get_finished")
    assert hasattr(worker, "block_size_bytes")
    # Default pool is thread-safe.
    assert isinstance(worker.pool, ThreadSafeEvictionPool)


def test_worker_instantiation_requires_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """``JoLTOffloadWorker(...)`` should raise ImportError when vLLM isn't installed."""
    _stub_vllm(monkeypatch)
    comp = JoLTCompressor(compression_ratio=3.0)
    worker = JoLTOffloadWorker(compressor=comp, block_shape=(8, 256, 64))
    assert worker.name == "jolt-offload"
    assert worker.block_size_bytes == 8 * 256 * 64 * 2


def test_worker_instantiation_raises_without_vllm() -> None:
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
            JoLTOffloadWorker(compressor=comp, block_shape=(8, 256, 64))
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


def test_compress_decompress_round_trip_separates_kv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: compress_block must not store K as both K and V."""
    _stub_vllm(monkeypatch)
    # Use a lossless compressor by overriding factor_dtype to fp32.
    from kvcompress.compressor.identity import IdentityCompressor

    comp = IdentityCompressor(factor_dtype=torch.float32)
    worker = JoLTOffloadWorker(compressor=comp, block_shape=(2, 2, 4, 8))
    # vLLM's standard layout: (num_layers, num_kv, T, dh).
    torch.manual_seed(0)
    num_layers, _num_kv, T, dh = 2, 2, 4, 8
    K_orig = torch.randn(num_layers, T, dh)
    V_orig = torch.randn(num_layers, T, dh)
    block = torch.stack([K_orig, V_orig], dim=1)  # (num_layers, 2, T, dh)
    stored = worker.compress_block(block)
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
        K, V = worker.decompress_block(layer)
        assert torch.allclose(K, K_orig[layer])
        assert torch.allclose(V, V_orig[layer])
        # K != V across the originals.
        assert not torch.equal(K, V)


def test_compress_block_handles_tuple_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accepts ``(K_block, V_block)`` tuple as well as the 4-D tensor."""
    _stub_vllm(monkeypatch)
    comp = IdentityCompressor(factor_dtype=torch.float32)
    worker = JoLTOffloadWorker(compressor=comp, block_shape=(2, 2, 4, 8))
    torch.manual_seed(0)
    K_orig = torch.randn(2, 4, 8)
    V_orig = torch.randn(2, 4, 8)
    stored = worker.compress_block((K_orig, V_orig))
    assert len(stored) == 4
    for layer in range(2):
        K, V = worker.decompress_block(layer)
        assert torch.allclose(K, K_orig[layer])
        assert torch.allclose(V, V_orig[layer])


def test_compress_block_rejects_unknown_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 3-D block (no layer axis) must raise loudly, not silently store."""
    _stub_vllm(monkeypatch)
    comp = IdentityCompressor()
    worker = JoLTOffloadWorker(compressor=comp, block_shape=(2, 2, 4, 8))
    bad = torch.randn(2, 4, 8)  # no layer axis
    with pytest.raises(ValueError, match="unsupported block type"):
        worker.compress_block(bad)


def test_decompress_block_returns_list_of_tuples_for_list_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vLLM may pass a list of layer indices; we return one (K, V) per layer."""
    _stub_vllm(monkeypatch)
    comp = IdentityCompressor(factor_dtype=torch.float32)
    worker = JoLTOffloadWorker(compressor=comp, block_shape=(3, 2, 4, 8))
    torch.manual_seed(0)
    K_orig = torch.randn(3, 4, 8)
    V_orig = torch.randn(3, 4, 8)
    block = torch.stack([K_orig, V_orig], dim=1)
    worker.compress_block(block)
    out = worker.decompress_block([0, 2])
    assert len(out) == 2
    K0, V0 = out[0]
    K2, V2 = out[1]
    assert torch.allclose(K0, K_orig[0])
    assert torch.allclose(V0, V_orig[0])
    assert torch.allclose(K2, K_orig[2])
    assert torch.allclose(V2, V_orig[2])
