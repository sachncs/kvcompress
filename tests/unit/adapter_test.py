"""Tests for the model-scoped Hugging Face adapter."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from kvfold.api import CompressionHandle, enable_compression


class FakeModel:
    """Minimal model-like object for adapter testing."""

    def __init__(self, model_type: str = "llama", has_cache_impl: bool = True) -> None:
        class Config:
            pass

        self.config = Config()
        self.config.model_type = model_type

        class GenConfig:
            pass

        self.generation_config = GenConfig()
        if has_cache_impl:
            self.generation_config.cache_implementation = None

    def _prepare_cache_for_generation(self, *_args: object, **_kwargs: object) -> object:
        """Provide the supported Transformers cache-construction hook."""
        from transformers.generation import utils

        return utils.DynamicCache


def test_enable_compression_on_fake_model() -> None:
    model = FakeModel("llama")
    handle = enable_compression(model, method="flash", ratio=2.0)
    try:
        assert isinstance(handle, CompressionHandle)
        assert handle.model is model
    finally:
        handle.disable()


def test_enable_compression_disables() -> None:
    model = FakeModel("mistral")
    handle = enable_compression(model, method="jolt", ratio=3.0)
    handle.disable()
    # No assertion on internal state; just that it doesn't raise.


def test_hf_adapter_exposes_polymorphic_lifecycle_state() -> None:
    from kvfold.adapter.huggingface import HF

    adapter = HF(model=FakeModel("llama"), method="jolt", ratio=2.0)
    assert adapter.state().attached is False
    with adapter as attached:
        assert attached.state().attached is True
        assert attached.state().name == "huggingface"
    assert adapter.state().attached is False


def test_hf_uses_model_scoped_cache_hook_when_available() -> None:
    import transformers.generation.utils as generation_utils

    model = FakeModel("llama")

    def prepare_cache(*_args: object, **_kwargs: object) -> object:
        return generation_utils.DynamicCache

    model._prepare_cache_for_generation = prepare_cache
    original = model._prepare_cache_for_generation
    handle = enable_compression(model, method="pass", ratio=1.0)
    try:
        scoped_class = model._prepare_cache_for_generation()
        assert scoped_class is not generation_utils.DynamicCache
    finally:
        handle.disable()
    assert model._prepare_cache_for_generation is original


def test_hf_cache_keeps_framework_layers_metadata_only() -> None:
    """The model-facing cache must not retain decompressed K/V tensors."""
    model = FakeModel("llama")
    handle = enable_compression(model, method="pass", target_memory=1.0)
    try:
        cache = model._prepare_cache_for_generation()()
        key = torch.randn(1, 2, 8, 4)
        value = torch.randn_like(key)
        cache.update(key, value, layer_idx=0)
        cache.update(key[..., :2, :], value[..., :2, :], layer_idx=0)
        assert cache.layers[0].keys is None
        assert cache.layers[0].values is None
        assert cache.get_seq_length(0) == 10
        assert cache.get_mask_sizes(torch.arange(2), 0) == (12, 0)
        assert cache.get_max_cache_shape(0) == -1
        assert cache.to_legacy_cache()[0][0].shape[-2] == 10

        cache.reorder_cache(torch.tensor([0]))
        cache.batch_repeat_interleave(1)
        cache.batch_select_indices(torch.tensor([0]))
        cache.crop(4)
        assert cache.get_seq_length(0) == 4
        cache.reset()
        assert len(cache) == 0
    finally:
        handle.disable()


def test_two_model_scoped_adapters_can_coexist() -> None:
    import transformers.generation.utils as generation_utils

    def add_hook(model: FakeModel) -> FakeModel:
        def prepare_cache(*_args: object, **_kwargs: object) -> object:
            return generation_utils.DynamicCache

        model._prepare_cache_for_generation = prepare_cache
        return model

    first = enable_compression(add_hook(FakeModel("llama")), method="pass", ratio=1.0)
    second = enable_compression(add_hook(FakeModel("mistral")), method="pass", ratio=1.0)
    try:
        assert first.is_active is True
        assert second.is_active is True
    finally:
        second.disable()
        first.disable()


def test_enable_compression_requires_target_or_ratio() -> None:
    model = FakeModel("llama")
    with pytest.raises(ValueError, match="target_memory"):
        enable_compression(model, method="jolt")


def test_models_without_scoped_cache_hook_fail_clearly() -> None:
    """Do not fall back to process-wide Transformers monkey-patching."""
    from kvfold.errors import TransformersCompatibilityError

    class UnsupportedModel:
        class Config:
            model_type = "unknown"

        class GenConfig:
            pass

        config = Config()
        generation_config = GenConfig()

    with pytest.raises(TransformersCompatibilityError, match="_prepare_cache"):
        enable_compression(UnsupportedModel(), method="pass", ratio=1.0)


def test_target_memory_parses() -> None:
    from kvfold.api import parse_target_memory

    assert parse_target_memory(0.25) == 4.0
    with pytest.raises(TypeError, match="numeric fraction"):
        parse_target_memory("25%")
    with pytest.raises(ValueError, match="fraction"):
        parse_target_memory(0)
    with pytest.raises(TypeError, match="numeric fraction"):
        parse_target_memory("150%")


def test_handle_stats_dict() -> None:
    model = FakeModel("qwen2")
    handle = enable_compression(model, method="flash", target_memory=0.33)
    try:
        d = handle.stats_dict()
        assert "compress_calls" in d
        assert "ratio" in d
    finally:
        handle.disable()


def test_compression_stats_snapshot_is_atomic() -> None:
    from kvfold.api import CompressionStats

    stats = CompressionStats()
    stats.record_compress(bytes_original=100, bytes_compressed=40, bytes_resident=48)
    stats.record_decompress()
    assert stats.snapshot() == {
        "compress_calls": 1,
        "decompress_calls": 1,
        "bytes_original": 100,
        "bytes_compressed": 40,
        "bytes_resident": 48,
        "ratio": 2.5,
        "memory_saved_bytes": 60,
    }


def test_compression_handle_references_cannot_be_reassigned() -> None:
    model = FakeModel("llama")
    handle = enable_compression(model, method="pass", ratio=1.0)
    try:
        with pytest.raises(dataclasses.FrozenInstanceError):
            handle.model = object()  # type: ignore[misc]
    finally:
        handle.disable()


def test_enable_compression_unknown_method_raises() -> None:
    from kvfold.errors import UnsupportedMethodError

    model = FakeModel("llama")
    with pytest.raises(UnsupportedMethodError, match="not-a-method"):
        enable_compression(model, method="not-a-method", ratio=2.0)


def test_invalid_cache_implementation_fails_early() -> None:
    with pytest.raises(ValueError, match="cache_implementation"):
        enable_compression(
            FakeModel("llama"), method="flash", ratio=2.0, cache_implementation="static"
        )


def test_missing_transformers_cache_api_fails_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers.cache_utils as cu

    from kvfold.adapter import huggingface as hf_module
    from kvfold.errors import TransformersCompatibilityError

    original = cu.DynamicCache
    monkeypatch.setattr(cu, "DynamicCache", None)
    adapter = hf_module.HF(model=FakeModel("llama"), method="flash", ratio=2.0)
    with pytest.raises(TransformersCompatibilityError, match="DynamicCache"):
        adapter.enable()
    assert not hasattr(adapter.model.generation_config, "cache_implementation")
    monkeypatch.setattr(cu, "DynamicCache", original)


def test_enable_compression_rejects_unknown_method_kwargs() -> None:
    """Typos in method-specific options fail instead of being ignored."""
    from kvfold.errors import MethodConfigError

    model = FakeModel("llama")
    with pytest.raises(MethodConfigError, match="not_a_real_option"):
        enable_compression(model, method="flash", ratio=2.0, not_a_real_option=True)


def test_enable_disables_leak_free_when_no_prior_attr() -> None:
    """``disable()`` must not leave ``cache_implementation="dynamic"``
    behind on a generation_config that didn't have the attribute.
    """
    model = FakeModel("llama", has_cache_impl=False)
    handle = enable_compression(model, method="flash", ratio=2.0)
    assert model.generation_config.cache_implementation == "dynamic"
    handle.disable()
    assert not hasattr(model.generation_config, "cache_implementation")


def test_enable_restores_prior_cache_implementation() -> None:
    """When the user pre-set ``cache_implementation``, ``disable()``
    must restore it exactly."""

    class GenConfig:
        cache_implementation = "static"

    class Model:
        class Config:
            model_type = "llama"

        config = Config()
        generation_config = GenConfig()

        def _prepare_cache_for_generation(self, *_args: object, **_kwargs: object) -> object:
            from transformers.generation import utils

            return utils.DynamicCache

    model = Model()
    handle = enable_compression(model, method="flash", ratio=2.0)
    assert model.generation_config.cache_implementation == "dynamic"
    handle.disable()
    assert model.generation_config.cache_implementation == "static"


def test_double_enable_raises_runtime_error() -> None:
    """A second ``enable_compression`` on the same model raises
    ``RuntimeError`` instead of silently no-oping.
    """
    model = FakeModel("llama")
    h1 = enable_compression(model, method="flash", ratio=2.0)
    try:
        with pytest.raises(RuntimeError, match="already"):
            enable_compression(model, method="jolt", ratio=3.0)
    finally:
        h1.disable()


def test_second_model_can_use_a_scoped_hook_concurrently() -> None:
    first = enable_compression(FakeModel("llama"), method="flash", ratio=2.0)
    second = enable_compression(FakeModel("mistral"), method="flash", ratio=2.0)
    try:
        assert first.is_active is True
        assert second.is_active is True
    finally:
        second.disable()
        first.disable()


def test_handle_is_active_property() -> None:
    model = FakeModel("llama")
    handle = enable_compression(model, method="flash", ratio=2.0)
    assert handle.is_active is True
    handle.disable()
    assert handle.is_active is False


def test_disable_is_idempotent() -> None:
    """Calling ``disable()`` twice doesn't raise."""
    model = FakeModel("llama")
    handle = enable_compression(model, method="flash", ratio=2.0)
    handle.disable()
    handle.disable()  # no-op
    assert handle.is_active is False
