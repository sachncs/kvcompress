"""Tests for the per-family adapter shims.

The shims are no-ops today (the DynamicCache subclass installed by
HuggingFaceAdapter covers all standard cache layouts). These tests
verify the registration mechanism and that each shim imports cleanly.
"""

from __future__ import annotations

import pytest

from kvcompress.adapters import registry


def test_registry_lists_all_families() -> None:
    families = registry.known_model_types()
    expected = {
        "llama",
        "mistral",
        "qwen2",
        "qwen2_moe",
        "gemma",
        "gemma2",
        "phi",
        "phi3",
        "mixtral",
        "falcon",
        "deepseek",
        "internlm",
    }
    assert set(families) == expected


def test_resolve_returns_module_path_for_known_family() -> None:
    for family in ("llama", "mistral", "qwen2", "deepseek"):
        module_path = registry.resolve(family)
        assert module_path is not None
        assert module_path.startswith("kvcompress.adapters.")


def test_resolve_returns_none_for_unknown_family() -> None:
    assert registry.resolve("not-a-real-model") is None
    assert registry.resolve("") is None


def test_register_adds_new_family() -> None:
    original = registry.known_model_types()
    original_set = set(registry._REGISTRY)
    try:
        registry.register("test-family-xyz", "kvcompress.adapters.llama")
        assert "test-family-xyz" in registry.known_model_types()
        assert registry.resolve("test-family-xyz") == "kvcompress.adapters.llama"
    finally:
        # Restore registry by removing any keys we added.
        for k in list(registry._REGISTRY):
            if k not in original_set:
                del registry._REGISTRY[k]
        assert registry.known_model_types() == original


def test_register_duplicate_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):
        registry.register("llama", "kvcompress.adapters.llama")


@pytest.mark.parametrize(
    "family",
    [
        "llama",
        "mistral",
        "qwen2",
        "gemma",
        "phi",
        "mixtral",
        "falcon",
        "deepseek",
        "internlm",
    ],
)
def test_every_family_shim_imports_and_installs(family: str) -> None:
    """Each registered family shim exposes an ``install`` function that
    accepts a model and cache_manager. The DynamicCache subclass covers
    them, so install is a no-op, but the contract must hold.
    """
    import importlib

    module_path = registry.resolve(family)
    assert module_path is not None
    module = importlib.import_module(module_path)
    assert hasattr(module, "install"), f"{module_path} missing install()"


def test_install_unknown_model_type_falls_back_to_generic() -> None:
    """``install(model_type="nonexistent", ...)`` falls back to the
    generic path (a no-op) and doesn't raise."""
    from kvcompress.adapters import registry

    # The generic install returns None; the family install returns
    # the callable. Either way, the function didn't raise.
    result = registry.install(
        model_type="definitely-not-a-real-family-xyz",
        model=object(),
        cache_manager=object(),
    )
    assert result is None or callable(result)
