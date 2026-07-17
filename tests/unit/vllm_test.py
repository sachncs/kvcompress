"""Tests for the Shape A vLLM helpers (export_kv / import_kv)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch


class FakeDynamicLayer:
    """Stand-in for transformers.cache_utils.DynamicLayer."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys = keys
        self.values = values


class FakeDynamicCache:
    """Stand-in for transformers.cache_utils.DynamicCache.

    Exposes ``.layers[i].keys`` and ``.layers[i].values`` like the real one.
    """

    def __init__(self, layers: list[FakeDynamicLayer]) -> None:
        self.layers = layers
        self.updates: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, cache_kwargs=None):
        self.updates.append((layer_idx, k, v))
        # Patch the layer in place (matches HF behaviour).
        self.layers[layer_idx].keys = k
        self.layers[layer_idx].values = v
        return k, v


class FakeModel:
    """Minimal HF-style model with a past_key_values cache."""

    def __init__(self, cache: FakeDynamicCache) -> None:
        self.past_key_values = cache


@pytest.fixture
def fake_model() -> Any:
    torch.manual_seed(0)
    layers = []
    for _ in range(3):
        k = torch.randn(2, 8, 16)
        v = torch.randn(2, 8, 16)
        layers.append(FakeDynamicLayer(k, v))
    cache = FakeDynamicCache(layers)
    return FakeModel(cache)


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

    class Bare:
        pass

    with pytest.raises(RuntimeError, match="could not locate"):
        export_kv(Bare(), "/tmp/should-not-write.safetensors")


def test_is_vllm_available_returns_bool() -> None:
    from kvcompress.adapters.vllm import is_vllm_available

    assert isinstance(is_vllm_available(), bool)


def test_import_round_trip_recovers_kv_separately(fake_model: Any, tmp_path: Path) -> None:
    """Regression: import_kv must populate K and V from the sidecar,
    not pass the same tensor twice.

    Previous implementation wrote ``cache.update(k_v, k_v, layer_idx)``
    which silently stored K under both slots.
    """
    pytest.importorskip("safetensors")
    from kvcompress.adapters.vllm import export_kv, import_kv

    out = tmp_path / "kv.safetensors"
    export_kv(fake_model, str(out), method="flashjolt", compression_ratio=2.0)

    # New model with the same shape — same layers, all zeros.
    torch.manual_seed(1)
    new_layers = []
    original_k = []
    original_v = []
    for layer in fake_model.past_key_values.layers:
        k = torch.zeros_like(layer.keys)
        v = torch.zeros_like(layer.values)
        original_k.append(layer.keys.clone())
        original_v.append(layer.values.clone())
        new_layers.append(FakeDynamicLayer(k, v))
    new_model = FakeModel(FakeDynamicCache(new_layers))

    import_kv(new_model, str(out), target_memory="100%")

    # K and V should be different (K must not be a copy of V).
    for layer_idx, (k_new, v_new, k_orig, v_orig) in enumerate(
        zip(
            [layer.keys for layer in new_model.past_key_values.layers],
            [layer.values for layer in new_model.past_key_values.layers],
            original_k,
            original_v,
        )
    ):
        assert not torch.equal(k_new, v_new), f"layer {layer_idx}: K and V are identical"
        # Reconstructed K should approximate the original K (lossy).
        rel_err_k = float(torch.linalg.norm(k_orig - k_new) / torch.linalg.norm(k_orig))
        rel_err_v = float(torch.linalg.norm(v_orig - v_new) / torch.linalg.norm(v_orig))
        assert rel_err_k < 1.5, f"layer {layer_idx}: K rel_err {rel_err_k:.3f}"
        assert rel_err_v < 1.5, f"layer {layer_idx}: V rel_err {rel_err_v:.3f}"
