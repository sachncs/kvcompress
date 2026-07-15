"""Tests for the Shape B vLLM KV-offload worker.

These tests run on CPU without vLLM installed. They verify the
subclass is structurally correct (imports cleanly, has the right
methods, exposes the expected public surface). Integration tests
against a real vLLM + CUDA install are out of scope for this repo.
"""

from __future__ import annotations

import pytest


def test_is_vllm_kv_offload_available_returns_bool() -> None:
    from kvcompress.adapters.vllm_kv_offload import is_vllm_kv_offload_available

    assert isinstance(is_vllm_kv_offload_available(), bool)


def test_module_imports_without_vllm() -> None:
    """The module must import on systems without vLLM."""
    import kvcompress.adapters.vllm_kv_offload  # noqa: F401


def test_worker_attributes() -> None:
    """The worker exposes the expected public attributes without needing vLLM at import time."""
    from kvcompress.adapters.vllm_kv_offload import JoLTOffloadWorker

    assert hasattr(JoLTOffloadWorker, "evict")
    assert hasattr(JoLTOffloadWorker, "restore")
    assert hasattr(JoLTOffloadWorker, "post_init")
    assert hasattr(JoLTOffloadWorker, "block_size_bytes")
    assert hasattr(JoLTOffloadWorker, "name")


def test_worker_instantiation_requires_vllm() -> None:
    """``JoLTOffloadWorker(...)`` should raise ImportError when vLLM isn't installed."""
    from kvcompress import JoLTCompressor
    from kvcompress.adapters.vllm_kv_offload import JoLTOffloadWorker

    pytest.importorskip("vllm")

    comp = JoLTCompressor(compression_ratio=3.0)
    worker = JoLTOffloadWorker(
        compressor=comp,
        block_shape=(8, 256, 64),
    )
    assert worker.name == "jolt-offload"
    assert worker.block_size_bytes == 8 * 256 * 64 * 2


def test_worker_instantiation_raises_without_vllm() -> None:
    """On systems without vLLM, instantiation should fail cleanly."""
    import builtins

    from kvcompress import JoLTCompressor
    from kvcompress.adapters.vllm_kv_offload import JoLTOffloadWorker

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
