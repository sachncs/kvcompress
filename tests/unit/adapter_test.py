"""Tests for the HF adapter and family registry."""

from __future__ import annotations

import pytest
import torch

from kvcompress.adapters.registry import (
    install,
    known_model_types,
    register,
    resolve,
)
from kvcompress.api import enable_compression, CompressionHandle


class _FakeModel:
    """Minimal model-like object for adapter testing."""

    def __init__(self, model_type: str = "llama") -> None:
        class _Config:
            pass

        self.config = _Config()
        self.config.model_type = model_type

        class _GenConfig:
            cache_implementation = None

        self.generation_config = _GenConfig()


def test_registry_known_types() -> None:
    types = known_model_types()
    assert "llama" in types
    assert "mistral" in types
    assert "qwen2" in types


def test_registry_resolve() -> None:
    assert resolve("llama") == "kvcompress.adapters.llama"
    assert resolve("unknown-type") is None


def test_registry_register_custom() -> None:
    register("custom-test", "kvcompress.adapters.llama")
    assert resolve("custom-test") == "kvcompress.adapters.llama"


def test_registry_register_duplicate_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register("llama", "kvcompress.adapters.llama")


def test_install_dispatches() -> None:
    model = _FakeModel("llama")
    from kvcompress.cache.manager import CacheManager
    from kvcompress.compressor.jolt import JoLTCompressor

    mgr = CacheManager(compressor=JoLTCompressor(compression_ratio=3.0))
    # Should not raise.
    install(model_type="llama", model=model, cache_manager=mgr)


def test_install_unknown_uses_generic() -> None:
    model = _FakeModel("nonexistent")
    from kvcompress.cache.manager import CacheManager
    from kvcompress.compressor.jolt import JoLTCompressor

    mgr = CacheManager(compressor=JoLTCompressor(compression_ratio=3.0))
    # Should not raise even though no shim exists.
    install(model_type="nonexistent", model=model, cache_manager=mgr)


def test_enable_compression_on_fake_model() -> None:
    model = _FakeModel("llama")
    handle = enable_compression(model, method="flashjolt", compression_ratio=2.0)
    try:
        assert isinstance(handle, CompressionHandle)
        assert handle.model is model
    finally:
        handle.disable()


def test_enable_compression_disables() -> None:
    model = _FakeModel("mistral")
    handle = enable_compression(model, method="jolt", compression_ratio=3.0)
    handle.disable()
    # No assertion on internal state; just that it doesn't raise.


def test_enable_compression_requires_target_or_ratio() -> None:
    model = _FakeModel("llama")
    with pytest.raises(ValueError, match="target_memory"):
        enable_compression(model, method="jolt")


def test_target_memory_parses() -> None:
    from kvcompress.api import _parse_target_memory

    assert _parse_target_memory("25%") == 4.0
    assert _parse_target_memory("50%") == 2.0
    assert _parse_target_memory(0.25) == 4.0
    with pytest.raises(ValueError):
        _parse_target_memory("abc")
    with pytest.raises(ValueError):
        _parse_target_memory(0)
    with pytest.raises(ValueError):
        _parse_target_memory("150%")


def test_handle_stats_dict() -> None:
    model = _FakeModel("qwen2")
    handle = enable_compression(model, method="flashjolt", target_memory="33%")
    try:
        d = handle.stats_dict()
        assert "compress_calls" in d
        assert "compression_ratio" in d
    finally:
        handle.disable()


def test_enable_compression_unknown_method_raises() -> None:
    model = _FakeModel("llama")
    with pytest.raises(NotImplementedError, match="not implemented"):
        enable_compression(model, method="not-a-method", compression_ratio=2.0)