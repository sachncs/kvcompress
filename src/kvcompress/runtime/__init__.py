"""Runtime helpers — scheduling, memory pools, profiling.

Imports are lazy so the package can be imported even when individual modules
are still stubbed out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kvcompress.runtime.memory import MemoryPool
    from kvcompress.runtime.profiler import CompressionProfiler

_LAZY_EXPORTS = {
    "MemoryPool": ("kvcompress.runtime.memory", "MemoryPool"),
    "CompressionProfiler": ("kvcompress.runtime.profiler", "CompressionProfiler"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        mod_name, attr = _LAZY_EXPORTS[name]
        module = importlib.import_module(mod_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'kvcompress.runtime' has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS)