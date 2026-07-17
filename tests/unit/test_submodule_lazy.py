"""Tests for the lazy-export ``__getattr__`` in submodule __init__.py.

The compressor / runtime subpackages export public symbols via
``__getattr__`` so they can be lazily imported. These tests verify
the lazy path resolves correctly and that unknown names raise
``AttributeError``.
"""

from __future__ import annotations

import pytest


def test_compressor_submodule_exports_jolt() -> None:
    from kvcompress.compressor import JoLTCompressor

    assert JoLTCompressor is not None
    assert JoLTCompressor.name == "jolt"


def test_compressor_submodule_exports_flashjolt() -> None:
    from kvcompress.compressor import FlashJoLTCompressor

    assert FlashJoLTCompressor is not None
    assert FlashJoLTCompressor.name == "flashjolt"


def test_compressor_submodule_exports_allocator_classes() -> None:
    from kvcompress.compressor import (
        Allocation,
        AllocationResult,
        Cell,
        JointAllocator,
    )

    assert JointAllocator.__name__ == "JointAllocator"
    assert Cell.__name__ == "Cell"
    assert Allocation.__name__ == "Allocation"
    assert AllocationResult.__name__ == "AllocationResult"


def test_compressor_submodule_exports_base_classes() -> None:
    from kvcompress.compressor import (
        CompressedPayload,
        CompressorStats,
        IdentityCompressor,
        IntQuantOnlyCompressor,
        KVCompressor,
        LowRankCompressor,
    )

    assert KVCompressor is not None
    assert CompressedPayload is not None
    assert CompressorStats is not None
    assert IdentityCompressor is not None
    assert IntQuantOnlyCompressor is not None
    assert LowRankCompressor is not None


def test_compressor_submodule_exports_dispatch_helpers() -> None:
    from kvcompress.compressor import METHODS, supported_methods

    methods = supported_methods()
    assert "jolt" in methods
    assert "flashjolt" in methods
    assert "lowrank" in methods
    assert "int2" in methods
    assert "identity" in methods
    # The METHODS dict has one entry per method.
    assert set(METHODS.keys()) == set(methods)


def test_compressor_submodule_unknown_name_raises() -> None:
    import kvcompress.compressor as sub

    with pytest.raises(AttributeError, match="no attribute"):
        _ = sub.NotARealThing


def test_runtime_submodule_exports_memory_pool() -> None:
    from kvcompress.runtime import MemoryPool

    assert MemoryPool.__name__ == "MemoryPool"


def test_runtime_submodule_exports_profiler() -> None:
    from kvcompress.runtime import CompressionProfiler

    assert CompressionProfiler.__name__ == "CompressionProfiler"


def test_runtime_submodule_unknown_name_raises() -> None:
    import kvcompress.runtime as sub

    with pytest.raises(AttributeError, match="no attribute"):
        _ = sub.PhantomSymbol
