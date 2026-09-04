"""kvfold — universal plug-and-play KV cache compression for decoder-only LLMs.

Implements the JoLT algorithm (partial Tucker decomposition + JL-rotated residual
+ joint Lagrangian allocation) and the Flash fast variant.

Public API:
    enable_compression(model, method=..., ...) — monkey-patch an HF model
    build_compressor(method)        — construct a Compressor directly
    supported_methods()             — tuple of every method name
    MethodName                      — Literal of supported method names
    CompressionHandle               — handle returned by enable_compression

Imports are lazy so the package can be imported even when individual modules
are still stubbed out during incremental development. The first attribute
access triggers the actual import via :func:`__getattr__` below.
"""

from __future__ import annotations

from typing import Any  # noqa: F401

__version__ = "0.2.0"

LAZY_EXPORTS = {
    "enable_compression": ("kvfold.api", "enable_compression"),
    "disable_compression": ("kvfold.api", "disable_compression"),
    "CompressionHandle": ("kvfold.api", "CompressionHandle"),
    "CompressionStats": ("kvfold.api", "CompressionStats"),
    "parse_target_memory": ("kvfold.api", "parse_target_memory"),
    "KVCompressor": ("kvfold.core.base", "KVCompressor"),
    "CompressorStats": ("kvfold.core.base", "CompressorStats"),
    "CompressedPayload": ("kvfold.core.base", "CompressedPayload"),
    "JoLTCompressor": ("kvfold.core.jolt", "JoLTCompressor"),
    "FlashJoLTCompressor": ("kvfold.core.flashjolt", "FlashJoLTCompressor"),
    "IdentityCompressor": ("kvfold.core.identity", "IdentityCompressor"),
    "LowRankCompressor": ("kvfold.core.lowrank", "LowRankCompressor"),
    "IntQuantOnlyCompressor": (
        "kvfold.core.quantization_only",
        "IntQuantOnlyCompressor",
    ),
    "JointAllocator": ("kvfold.core.allocator", "JointAllocator"),
    "Allocation": ("kvfold.core.allocator", "Allocation"),
    "Cell": ("kvfold.core.allocator", "Cell"),
    "build_compressor": ("kvfold.api", "build_compressor"),
    "supported_methods": ("kvfold.api", "supported_methods"),
    "CompressedKVCache": ("kvfold.cache.compress", "CompressedKVCache"),
    "CacheManager": ("kvfold.cache.manager", "CacheManager"),
    "CompressionMetadata": ("kvfold.cache.metadata", "CompressionMetadata"),
    "LayerCompression": ("kvfold.cache.metadata", "LayerCompression"),
}


def __getattr__(name: str) -> Any:
    if name in LAZY_EXPORTS:
        import importlib

        mod_name, attr = LAZY_EXPORTS[name]
        module = importlib.import_module(mod_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'kvfold' has no attribute {name!r}")


_ = __getattr__("build_compressor")  # initialise the import-side-effect module
del _


__all__ = ["__version__", *LAZY_EXPORTS]
