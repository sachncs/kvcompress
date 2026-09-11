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

import kvfold.core.builtins  # noqa: F401 — populates the compressor dispatcher registry

__version__ = "0.2.0"

LAZY_EXPORTS = {
    "enable_compression": ("kvfold.api", "enable_compression"),
    "disable_compression": ("kvfold.api", "disable_compression"),
    "CompressionHandle": ("kvfold.api", "CompressionHandle"),
    "CompressionStats": ("kvfold.api", "CompressionStats"),
    "parse_target_memory": ("kvfold.api", "parse_target_memory"),
    "build_compressor": ("kvfold.api", "build_compressor"),
    "supported_methods": ("kvfold.api", "supported_methods"),
    "MethodName": ("kvfold.api", "MethodName"),
    "Jolt": ("kvfold.core.jolt", "Jolt"),
    "Flash": ("kvfold.core.flash", "Flash"),
    "Low": ("kvfold.core.low", "Low"),
    "Pass": ("kvfold.core.identity", "Pass"),
    "Float8": ("kvfold.core.float8", "Float8"),
    "FloatCast": ("kvfold.core.float_cast", "FloatCast"),
    "IntQuant": ("kvfold.core.int_quant", "IntQuant"),
    "Compressor": ("kvfold.core.base", "Compressor"),
    "Payload": ("kvfold.core.base", "Payload"),
    "Stats": ("kvfold.core.base", "Stats"),
    "Bisect": ("kvfold.core.budget", "Bisect"),
    "Greedy": ("kvfold.core.budget", "Greedy"),
    "Plan": ("kvfold.core.budget", "Plan"),
    "Pick": ("kvfold.core.budget", "Pick"),
    "Cell": ("kvfold.core.budget", "Cell"),
    "Allocator": ("kvfold.core.budget", "Allocator"),
    "AllocatorRegistry": ("kvfold.core.budget", "AllocatorRegistry"),
    "Tucker": ("kvfold.core.tucker", "Tucker"),
    "TuckerDecomposer": ("kvfold.core.tucker", "TuckerDecomposer"),
    "TuckerReconstructor": ("kvfold.core.tucker", "TuckerReconstructor"),
    "TuckerBackendRegistry": ("kvfold.core.tucker", "TuckerBackendRegistry"),
    "HoSVDDecomposer": ("kvfold.core.tucker", "HoSVDDecomposer"),
    "TorchReconstructor": ("kvfold.core.tucker", "TorchReconstructor"),
    "Projector": ("kvfold.core.jl", "Projector"),
    "Gaussian": ("kvfold.core.jl", "Gaussian"),
    "Rademacher": ("kvfold.core.jl", "Rademacher"),
    "Sparse": ("kvfold.core.jl", "Sparse"),
    "Projection": ("kvfold.core.jl", "Projection"),
    "ProjectionCache": ("kvfold.core.jl", "ProjectionCache"),
    "Decomposer": ("kvfold.core.svd", "Decomposer"),
    "Exact": ("kvfold.core.svd", "Exact"),
    "Randomized": ("kvfold.core.svd", "Randomized"),
    "Decomposition": ("kvfold.core.svd", "Decomposition"),
    "Quantizer": ("kvfold.core.quant", "Quantizer"),
    "IntQuantImpl": ("kvfold.core.quant", "IntQuant"),
    "FloatCastQuantizer": ("kvfold.core.quant", "FloatCastQuantizer"),
    "bit_packing_signed": ("kvfold.core.quant", "bit_packing_signed"),
    "bit_unpacking_signed": ("kvfold.core.quant", "bit_unpacking_signed"),
    "quantize_tensor": ("kvfold.core.quant", "quantize_tensor"),
    "dequantize_tensor": ("kvfold.core.quant", "dequantize_tensor"),
    "estimate_int_bytes": ("kvfold.core.quant", "estimate_int_bytes"),
    "Residual": ("kvfold.core.residual", "Residual"),
    "Family": ("kvfold.adapter.registry", "Family"),
    "FamilyRegistry": ("kvfold.adapter.registry", "FamilyRegistry"),
    "HF": ("kvfold.adapter.huggingface", "HF"),
    "Cache": ("kvfold.store.compress", "Cache"),
    "Pool": ("kvfold.store.manager", "Pool"),
    "Meta": ("kvfold.store.metadata", "Meta"),
    "LayerMeta": ("kvfold.store.metadata", "LayerMeta"),
    "MethodConfig": ("kvfold.config", "MethodConfig"),
    "MethodConfigRegistry": ("kvfold.config", "MethodConfigRegistry"),
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


__all__ = ["__version__", *LAZY_EXPORTS]
