"""Compressor implementations and core tensor algebra.

Public symbols are exposed lazily through :func:`__getattr__` so this package
can be imported even when individual modules are still stubbed out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kvcompress.compressor.allocator import (
        Allocation,
        Cell,
        JointAllocator,
    )
    from kvcompress.compressor.base import (
        CompressorStats,
        KVCompressor,
    )
    from kvcompress.compressor.flashjolt import FlashJoLTCompressor
    from kvcompress.compressor.jolt import JoLTCompressor

_LAZY_EXPORTS = {
    "Allocation": ("kvcompress.compressor.allocator", "Allocation"),
    "Cell": ("kvcompress.compressor.allocator", "Cell"),
    "JointAllocator": ("kvcompress.compressor.allocator", "JointAllocator"),
    "CompressorStats": ("kvcompress.compressor.base", "CompressorStats"),
    "KVCompressor": ("kvcompress.compressor.base", "KVCompressor"),
    "FlashJoLTCompressor": ("kvcompress.compressor.flashjolt", "FlashJoLTCompressor"),
    "JoLTCompressor": ("kvcompress.compressor.jolt", "JoLTCompressor"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        mod_name, attr = _LAZY_EXPORTS[name]
        module = importlib.import_module(mod_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'kvcompress.compressor' has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS)