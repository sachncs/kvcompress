"""Tests for the lazy-export ``__getattr__`` in submodule __init__.py.

The compressor / runtime subpackages export public symbols via
``__getattr__`` so they can be lazily imported. These tests verify
the lazy path resolves correctly and that unknown names raise
``AttributeError``.
"""

from __future__ import annotations

import pytest


def test_core_submodule_exports_jolt() -> None:
    from kvfold.core import Jolt

    assert Jolt is not None
    assert Jolt.method == "jolt"


def test_core_submodule_exports_flash() -> None:
    from kvfold.core import Flash

    assert Flash is not None
    assert Flash.method == "flash"


def test_core_submodule_exports_budget_classes() -> None:
    from kvfold.core import Cell, Bisect

    assert Bisect.__name__ == "Bisect"
    assert Cell.__name__ == "Cell"


def test_core_submodule_exports_base_classes() -> None:
    from kvfold.core import Compressor, CompressorRegistry, Projector

    assert Compressor is not None
    assert CompressorRegistry is not None
    assert Projector is not None


def test_core_submodule_exports_dispatch_helpers() -> None:
    from kvfold.core import CONFIG_REGISTRY

    names = CONFIG_REGISTRY.names()
    assert "jolt" in names
    assert "flash" in names
    assert "low" in names
    assert "int2" in names
    assert "pass" in names


def test_core_submodule_unknown_name_raises() -> None:
    import kvfold.core as sub

    with pytest.raises(AttributeError, match="no attribute"):
        _ = sub.NotARealThing


def test_runtime_submodule_exports_memory_pool() -> None:
    from kvfold.runtime import Pool

    assert Pool.__name__ == "Pool"


def test_runtime_submodule_exports_profiler() -> None:
    from kvfold.runtime import Profile

    assert Profile.__name__ == "Profile"


def test_runtime_submodule_exports_seed() -> None:
    from kvfold.runtime.seed import Seed, generator

    assert Seed is not None
    assert generator is not None


def test_runtime_submodule_unknown_name_raises() -> None:
    import kvfold.runtime as sub

    with pytest.raises(AttributeError, match="no attribute"):
        _ = sub.PhantomSymbol
