"""Tests for the Shape A vLLM helpers (export_kv / import_kv)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch


class _FakeDynamicLayer:
    """Stand-in for transformers.cache_utils.DynamicLayer."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys = keys
        self.values = values


class _FakeDynamicCache:
    """Stand-in for transformers.cache_utils.DynamicCache.

    Exposes ``.layers[i].keys`` and ``.layers[i].values`` like the real one.
    """

    def __init__(self, layers: list[_FakeDynamicLayer]) -> None:
        self.layers = layers
        self._updates: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, cache_kwargs=None):
        self._updates.append((layer_idx, k, v))
        # Patch the layer in place (matches HF behaviour).
        self.layers[layer_idx].keys = k
        self.layers[layer_idx].values = v
        return k, v


class _FakeModel:
    """Minimal HF-style model with a past_key_values cache."""

    def __init__(self, cache: _FakeDynamicCache) -> None:
        self.past_key_values = cache


@pytest.fixture
def fake_model() -> Any:
    torch.manual_seed(0)
    layers = []
    for _ in range(3):
        k = torch.randn(2, 8, 16)
        v = torch.randn(2, 8, 16)
        layers.append(_FakeDynamicLayer(k, v))
    cache = _FakeDynamicCache(layers)
    return _FakeModel(cache)


def test_export_writes_safetensors(fake_model: Any, tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    from kvcompress.adapters.vllm import export_kv

    out = tmp_path / "kv.safetensors"
    meta = export_kv(
        fake_model,
        str(out),
        method="flashjolt",
        compression_ratio=2.0,
    )

    assert out.exists()
    assert (tmp_path / "kv.safetensors.meta.json").exists()
    assert len(meta.layers) >= 1
    # Each layer should have a K and V entry.
    kinds = {entry.kind for entry in meta.layers}
    assert kinds == {"key", "value"}


def test_export_resolves_compressor_correctly(fake_model: Any, tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    from kvcompress.adapters.vllm import export_kv

    out = tmp_path / "kv2.safetensors"
    meta = export_kv(fake_model, str(out), method="jolt", compression_ratio=2.0)
    assert meta.method in ("jolt", "flashjolt")  # JoLT method string
    assert len(meta.layers) > 0


def test_export_uses_passed_compressor(fake_model: Any, tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    from kvcompress import JoLTCompressor
    from kvcompress.adapters.vllm import export_kv

    comp = JoLTCompressor(compression_ratio=4.0)
    out = tmp_path / "kv3.safetensors"
    meta = export_kv(fake_model, str(out), compressor=comp)
    # Allocation decisions should reflect the 4x target.
    assert meta.method == "jolt"


def test_export_requires_cache() -> None:
    from kvcompress.adapters.vllm import export_kv

    class _Bare:
        pass

    with pytest.raises(RuntimeError, match="could not locate"):
        export_kv(_Bare(), "/tmp/should-not-write.safetensors")


def test_is_vllm_available_returns_bool() -> None:
    from kvcompress.adapters.vllm import is_vllm_available

    assert isinstance(is_vllm_available(), bool)
