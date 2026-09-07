"""Tests for the per-family adapter shims.

The shims are no-op :class:`Family` subclasses today (the
DynamicCache subclass installed by the HF adapter covers all standard
cache layouts). These tests verify the registration mechanism and that
every built-in family loads cleanly.
"""

from __future__ import annotations

import pytest

from kvfold.adapter import registry
from kvfold.adapter.registry import (
    Family,
    FamilyRegistry,
    REGISTRY,
    install,
    known_model_types,
    resolve,
)


EXPECTED_FAMILIES = {
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


def test_registry_lists_all_families() -> None:
    assert set(known_model_types()) == EXPECTED_FAMILIES


@pytest.mark.parametrize("family", list(EXPECTED_FAMILIES))
def test_resolve_returns_family_for_known(family: str) -> None:
    f = resolve(family)
    assert f is not None
    assert f.name == family


def test_resolve_returns_none_for_unknown_family() -> None:
    assert resolve("not-a-real-model") is None
    assert resolve("") is None


def test_register_adds_new_family() -> None:
    class _CustomFamily(Family):
        name = "test-family-xyz"

        def install(self, model, pool):
            return None

    original_count = len(REGISTRY.entries)
    REGISTRY.register(_CustomFamily)
    try:
        assert "test-family-xyz" in known_model_types()
        assert resolve("test-family-xyz") is not None
    finally:
        REGISTRY.entries.pop("test-family-xyz", None)
    assert len(REGISTRY.entries) == original_count


def test_register_duplicate_raises() -> None:
    from kvfold.adapter.registry import Llama
    with pytest.raises(ValueError, match="already registered"):
        REGISTRY.register(Llama)


def test_install_returns_none_for_no_op_family() -> None:
    """Each NoOpFamily's install() returns None."""
    for family in EXPECTED_FAMILIES:
        f = resolve(family)
        result = f.install(object(), object())
        assert result is None


def test_install_module_function_dispatches_to_family() -> None:
    """The module-level install() forwards to the registered family."""
    result = install(object(), object(), model_type="llama")
    assert result is None


def test_install_module_function_unknown_returns_none() -> None:
    result = install(object(), object(), model_type="not-a-real-family-xyz")
    assert result is None


def test_registry_is_family_registry_instance() -> None:
    assert isinstance(REGISTRY, FamilyRegistry)
