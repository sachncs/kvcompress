"""Core compression algorithms and supporting math.

Modules are imported lazily so the package can be loaded even when an
optional dependency (vLLM, Triton) is missing. Concrete compressors are
registered with the :class:`CompressorRegistry` on import.
"""

from __future__ import annotations

from typing import Any

from kvfold.config import REGISTRY as CONFIG_REGISTRY

LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Compressor": ("kvfold.core.base", "Compressor"),
    "Payload": ("kvfold.core.base", "Payload"),
    "Stats": ("kvfold.core.base", "Stats"),
    "CompressorRegistry": ("kvfold.core.dispatch", "CompressorRegistry"),
    "Jolt": ("kvfold.core.jolt", "Jolt"),
    "Flash": ("kvfold.core.flash", "Flash"),
    "Low": ("kvfold.core.low", "Low"),
    "Pass": ("kvfold.core.identity", "Pass"),
    "IntQuant": ("kvfold.core.int_quant", "IntQuant"),
    "Float8": ("kvfold.core.float8", "Float8"),
    "FloatCast": ("kvfold.core.quant", "FloatCast"),
    "Tucker": ("kvfold.core.tucker", "Tucker"),
    "TuckerDecomposer": ("kvfold.core.tucker", "TuckerDecomposer"),
    "TuckerReconstructor": ("kvfold.core.tucker", "TuckerReconstructor"),
    "Residual": ("kvfold.core.residual", "Residual"),
    "Projector": ("kvfold.core.jl", "Projector"),
    "Decomposer": ("kvfold.core.svd", "Decomposer"),
    "Quantizer": ("kvfold.core.quant", "Quantizer"),
    "RankStrategy": ("kvfold.core.rank", "RankStrategy"),
    "Allocator": ("kvfold.core.budget", "Allocator"),
    "Bisect": ("kvfold.core.budget", "Bisect"),
    "Greedy": ("kvfold.core.budget", "Greedy"),
    "Pick": ("kvfold.core.budget", "Pick"),
    "Plan": ("kvfold.core.budget", "Plan"),
    "Cell": ("kvfold.core.budget", "Cell"),
}


def __getattr__(name: str) -> Any:
    target = LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'kvfold.core' has no attribute {name!r}")
    module_path, attr = target
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, attr)
    globals()[name] = value
    return value


__all__ = list(LAZY_EXPORTS) + ["CONFIG_REGISTRY"]
