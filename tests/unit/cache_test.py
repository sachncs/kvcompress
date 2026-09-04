"""Tests for the CompressedKVCache and CacheManager."""

from __future__ import annotations

import pytest
import torch

from kvfold.store.compress import CompressedKVCache
from kvfold.store.manager import CacheManager
from kvfold.store.metadata import CompressionMetadata, LayerCompression
from kvfold.core.base import (
    Payload,
    Stats,
    Compressor,
)


class Identity(Compressor):
    """Test compressor that stores K and V as fp16 with no actual compression."""

    name = "identity-test"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def compress(self, key: torch.Tensor, value: torch.Tensor):
        kp = Payload(
            method="identity-test",
            shape=tuple(key.shape),
            dtype=key.dtype,
            metadata={"r_token": 0, "r_feature": 0, "bits": 0},
            data={"value": key.to(torch.float16)},
            stats=Stats(bytes_original=key.numel() * key.element_size()),
        )
        vp = Payload(
            method="identity-test",
            shape=tuple(value.shape),
            dtype=value.dtype,
            metadata={"r_token": 0, "r_feature": 0, "bits": 0},
            data={"value": value.to(torch.float16)},
            stats=Stats(bytes_original=value.numel() * value.element_size()),
        )
        self.calls += 1
        return kp, vp

    def decompress(self, kp: Payload, vp: Payload):
        return kp.data["value"].to(kp.dtype), vp.data["value"].to(vp.dtype)


@pytest.fixture
def comp() -> Identity:
    return Identity()


@pytest.fixture
def kv() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(2, 8, 16), torch.randn(2, 8, 16)


def test_store_and_retrieve(comp: Identity, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    cache = CompressedKVCache(compressor=comp)
    k, v = kv
    cache.store(layer=0, key=k, value=v)
    k_hat, v_hat = cache.retrieve(0)
    assert k_hat.shape == k.shape
    assert v_hat.shape == v.shape


def test_store_4d_input_normalizes(comp: Identity) -> None:
    """HF-style (B, n_kv, T, dh) input is reshaped to (B·n_kv, T, dh)."""
    cache = CompressedKVCache(compressor=comp)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    cache.store(layer=0, key=k, value=v)
    k_hat, _ = cache.retrieve(0)
    # Retrieved as (2*4, 8, 16).
    assert k_hat.shape == (8, 8, 16)


def test_clear(comp: Identity, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    cache = CompressedKVCache(compressor=comp)
    cache.store(layer=0, key=kv[0], value=kv[1])
    cache.clear()
    assert 0 not in cache
    assert len(cache) == 0


def test_evict(comp: Identity, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    cache = CompressedKVCache(compressor=comp)
    cache.store(layer=0, key=kv[0], value=kv[1])
    cache.store(layer=1, key=kv[0], value=kv[1])
    cache.evict_layer(0)
    assert 0 not in cache
    assert 1 in cache


def test_max_layers_lru(comp: Identity, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    cache = CompressedKVCache(compressor=comp, max_layers=2)
    cache.store(layer=0, key=kv[0], value=kv[1])
    cache.store(layer=1, key=kv[0], value=kv[1])
    cache.store(layer=2, key=kv[0], value=kv[1])  # evicts layer 0
    assert 0 not in cache
    assert 1 in cache
    assert 2 in cache


def test_memory_used(comp: Identity, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    cache = CompressedKVCache(compressor=comp)
    cache.store(layer=0, key=kv[0], value=kv[1])
    used = cache.memory_used()
    original = cache.memory_original()
    assert used > 0
    assert original > 0
    # Identity stores as fp16, so used ≈ original / 2.
    assert used < original


def test_stats(comp: Identity, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    cache = CompressedKVCache(compressor=comp)
    cache.store(layer=0, key=kv[0], value=kv[1])
    s = cache.stats()
    assert s["n_layers"] == 1
    assert "bytes_original" in s
    assert "bytes_compressed" in s
    assert s["method"] == "identity-test"


def test_metadata(comp: Identity, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    cache = CompressedKVCache(compressor=comp)
    cache.store(layer=0, key=kv[0], value=kv[1])
    meta = cache.metadata()
    assert isinstance(meta, CompressionMetadata)
    assert len(meta.layers) == 2  # K and V entries


def test_layers_iterator(comp: Identity, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    cache = CompressedKVCache(compressor=comp)
    for i in range(3):
        cache.store(layer=i, key=kv[0], value=kv[1])
    layers = list(cache.layers())
    assert layers == [0, 1, 2]


def test_payload_access(comp: Identity, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    cache = CompressedKVCache(compressor=comp)
    cache.store(layer=0, key=kv[0], value=kv[1])
    p = cache.payload(0, "key")
    assert p.method == "identity-test"
    with pytest.raises(ValueError, match="kind"):
        cache.payload(0, "weird")


def test_metadata_layer_roundtrip() -> None:
    meta = CompressionMetadata(
        method="jolt",
        dtype="float16",
        layer_groups=1,
        bits_allowed=(0, 2, 4, 8),
    )
    meta.add_layer(
        LayerCompression(
            layer=0,
            kind="key",
            m=4,
            tokens=64,
            dh=16,
            r_token=8,
            r_feature=4,
            bits=4,
            bytes_original=8192,
            bytes_compressed=2048,
        )
    )
    d = meta.to_dict()
    meta2 = CompressionMetadata.from_dict(d)
    assert meta2.method == "jolt"
    assert meta2.layer(0).r_token == 8


def test_cache_manager(comp: Identity, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    mgr = CacheManager(compressor=comp)
    mgr.store(0, kv[0], kv[1])
    k, v = mgr.retrieve(0)
    assert k.shape == kv[0].shape
    assert 0 in mgr
    mgr.clear()
    assert len(mgr) == 0


def test_cache_manager_evict(comp: Identity, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    mgr = CacheManager(compressor=comp)
    mgr.store(0, kv[0], kv[1])
    mgr.evict(0)
    assert 0 not in mgr


def test_cache_manager_stats(comp: Identity, kv: tuple[torch.Tensor, torch.Tensor]) -> None:
    mgr = CacheManager(compressor=comp)
    mgr.store(0, kv[0], kv[1])
    s = mgr.stats()
    assert "live_layers" in s
    assert 0 in s["live_layers"]
