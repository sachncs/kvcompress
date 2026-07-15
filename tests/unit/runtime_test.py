"""Tests for runtime helpers (memory pool, profiler)."""

from __future__ import annotations

import pytest
import torch

from kvcompress.runtime.memory import MemoryPool
from kvcompress.runtime.profiler import CompressionProfiler


def test_memory_pool_acquire() -> None:
    pool = MemoryPool()
    t = pool.acquire((4, 8), dtype=torch.float32)
    assert t.shape == (4, 8)
    assert t.dtype == torch.float32


def test_memory_pool_reuse() -> None:
    pool = MemoryPool()
    t1 = pool.acquire((4, 8), dtype=torch.float32)
    pool.release(t1)
    t2 = pool.acquire((4, 8), dtype=torch.float32)
    # Should reuse the same tensor object.
    assert t1 is t2


def test_memory_pool_separates_keys() -> None:
    pool = MemoryPool()
    t1 = pool.acquire((4, 8), dtype=torch.float32)
    t2 = pool.acquire((4, 4), dtype=torch.float32)
    assert t1.shape != t2.shape


def test_memory_pool_caps_per_key() -> None:
    pool = MemoryPool(max_per_key=2)
    ts = [pool.acquire((4,)) for _ in range(5)]
    for t in ts:
        pool.release(t)
    s = pool.stats()
    assert s["total_buffers"] == 2


def test_memory_pool_clear() -> None:
    pool = MemoryPool()
    t = pool.acquire((4,))
    pool.release(t)
    pool.clear()
    assert pool.stats()["total_buffers"] == 0


def test_profiler_records() -> None:
    p = CompressionProfiler()
    with p.record("compress", bytes_in=100):
        pass
    s = p.summary()
    assert "compress" in s
    assert s["compress"]["count"] == 1
    assert s["compress"]["bytes_in"] == 100


def test_profiler_aggregate() -> None:
    p = CompressionProfiler()
    for _ in range(3):
        with p.record("op", bytes_in=10):
            pass
    s = p.summary()
    assert s["op"]["count"] == 3
    assert s["op"]["bytes_in"] == 30


def test_profiler_disabled() -> None:
    p = CompressionProfiler()
    p._enabled = False
    with p.record("op"):
        pass
    assert len(p.records) == 0


def test_profiler_reset() -> None:
    p = CompressionProfiler()
    with p.record("op"):
        pass
    p.reset()
    assert len(p.records) == 0