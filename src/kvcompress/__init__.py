"""kvcompress — universal plug-and-play KV cache compression for decoder-only LLMs.

Implements the JoLT algorithm (partial Tucker decomposition + JL-rotated residual
+ joint Lagrangian allocation) and the FlashJoLT fast variant.

Public API:
    enable_compression(model, method=..., ...) — monkey-patch an HF model
    KVCompressor           — abstract base for all compressors
    JoLTCompressor         — paper-faithful JoLT
    FlashJoLTCompressor    — randomized-SVD JoLT
    CompressedKVCache      — layer-indexed compressed cache
    CacheManager           — high-level cache orchestration

Imports are lazy so the package can be imported even when individual modules
are still stubbed out during incremental development.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "enable_compression": ("kvcompress.api", "enable_compression"),
    "disable_compression": ("kvcompress.api", "disable_compression"),
    "CompressionHandle": ("kvcompress.api", "CompressionHandle"),
    "CompressionStats": ("kvcompress.api", "CompressionStats"),
    "KVCompressor": ("kvcompress.compressor.base", "KVCompressor"),
    "CompressorStats": ("kvcompress.compressor.base", "CompressorStats"),
    "CompressedPayload": ("kvcompress.compressor.base", "CompressedPayload"),
    "JoLTCompressor": ("kvcompress.compressor.jolt", "JoLTCompressor"),
    "FlashJoLTCompressor": ("kvcompress.compressor.flashjolt", "FlashJoLTCompressor"),
    "JointAllocator": ("kvcompress.compressor.allocator", "JointAllocator"),
    "Allocation": ("kvcompress.compressor.allocator", "Allocation"),
    "Cell": ("kvcompress.compressor.allocator", "Cell"),
    "CompressedKVCache": ("kvcompress.cache.compress", "CompressedKVCache"),
    "CacheManager": ("kvcompress.cache.manager", "CacheManager"),
    "CompressionMetadata": ("kvcompress.cache.metadata", "CompressionMetadata"),
    "LayerCompression": ("kvcompress.cache.metadata", "LayerCompression"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        mod_name, attr = _LAZY_EXPORTS[name]
        module = importlib.import_module(mod_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'kvcompress' has no attribute {name!r}")


__all__ = ["__version__", *_LAZY_EXPORTS]