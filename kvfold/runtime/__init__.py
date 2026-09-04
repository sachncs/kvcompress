"""Runtime helpers — scheduling, memory pools, profiling.

Imports are lazy so the package can be imported even when individual modules
are still stubbed out.

* :mod:`.memory` defines :class:`~kvfold.runtime.memory.Pool`,
  a small object pool that reuses contiguous tensors across compress /
  decompress calls to reduce allocator pressure during long-context
  generation.
* :mod:`.profiler` defines :class:`~kvfold.runtime.profiler.Profile`,
  a context-manager-style timer used by the benchmark suite.
"""

from __future__ import annotations

from typing import Any  # noqa: F401

if False:  # TYPE_CHECKING
    from kvfold.runtime.pool import Pool  # noqa: F401
    from kvfold.runtime.profile import Profile  # noqa: F401

LAZY_EXPORTS = {
    "Pool": ("kvfold.runtime.memory", "Pool"),
    "Profile": ("kvfold.runtime.profiler", "Profile"),
}


def __getattr__(name: str) -> Any:
    if name in LAZY_EXPORTS:
        import importlib

        mod_name, attr = LAZY_EXPORTS[name]
        module = importlib.import_module(mod_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'kvfold.runtime' has no attribute {name!r}")


__all__ = list(LAZY_EXPORTS)
