"""Authoritative tests for framework-backed compressed cache residency."""

from __future__ import annotations

import pytest
import torch

from kvfold import enable_compression


class _CacheHookModel:
    """Minimal model exposing the supported Transformers cache hook."""

    def _prepare_cache_for_generation(self, *_args: object, **_kwargs: object) -> type:
        from transformers.generation import utils

        return utils.DynamicCache


@pytest.mark.integration
@pytest.mark.parametrize("device_type", ["cpu", "mps"])
def test_framework_cache_has_no_uncompressed_kv_residency(device_type: str) -> None:
    """The framework cache itself must not retain the complete K/V tensors."""
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple MPS is not available")

    device = torch.device(device_type)
    model = _CacheHookModel()
    handle = enable_compression(
        model,
        method="int4",
        target_memory=0.25,
        device=device,
    )
    try:
        cache_cls = model._prepare_cache_for_generation()
        compressed_cache = cache_cls()
        key = torch.randn(1, 4, 128, 32, device=device)
        value = torch.randn_like(key)

        compressed_cache.update(key, value, layer_idx=0)

        assert compressed_cache.layers[0].keys is None
        assert compressed_cache.layers[0].values is None
        assert handle.adapter.manager is not None
        assert handle.adapter.manager.memory_resident() < (
            key.untyped_storage().nbytes() + value.untyped_storage().nbytes()
        )
        restored_key, restored_value = compressed_cache[0]
        assert restored_key.shape == key.shape
        assert restored_value.shape == value.shape
        assert restored_key.device.type == device_type
        assert compressed_cache.get_seq_length(0) == key.shape[-2]
    finally:
        handle.disable()
