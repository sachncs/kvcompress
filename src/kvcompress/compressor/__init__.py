"""Compressor implementations and core tensor algebra.

Public symbols are exposed lazily through :func:`__getattr__` so this package
can be imported even when individual modules are still stubbed out.

Subpackages and modules:

* :mod:`.base` defines the :class:`KVCompressor` ABC that every concrete
  compressor subclasses.
* :mod:`.jolt` is the paper-faithful JoLT implementation.
* :mod:`.flashjolt` is the randomised-SVD fast variant.
* :mod:`.lowrank` and :mod:`.quantization_only` are ablation baselines.
* :mod:`.identity` is a passthrough compressor used for benchmarking.
* :mod:`.allocator` holds the joint Lagrangian allocator and the greedy
  ablation allocator.
* :mod:`.tucker` implements the partial Tucker ST-HOSVD used as JoLT's
  low-rank backbone.
* :mod:`.svd` is the unified exact + randomised SVD class.
* :mod:`.jl` is the Johnson-Lindenstrauss projection used to rotate the
  residual before quantisation.
* :mod:`.quantization` provides the FP16/BF16/FP8/INT2/4/8 quantizers.
* :mod:`.residual` packages the JL-rotated residual into a serialisable
  payload.

Why lazy imports: during incremental development we want the package to
remain importable even when a single submodule is mid-edit. Lazy loading
makes the public surface robust to that.
"""

from __future__ import annotations

from typing import Any  # noqa: F401

if False:  # TYPE_CHECKING
    from kvcompress.compressor.allocator import (  # noqa: F401
        Allocation,
        Cell,
        JointAllocator,
    )
    from kvcompress.compressor.base import (  # noqa: F401
        CompressorStats,
        KVCompressor,
    )
    from kvcompress.compressor.flashjolt import FlashJoLTCompressor  # noqa: F401
    from kvcompress.compressor.jolt import JoLTCompressor  # noqa: F401

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
