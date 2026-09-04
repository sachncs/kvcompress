"""Optional Triton kernels.

Imported only when :mod:`triton` is installed. On systems without
Triton, the public functions in this module transparently fall back to
PyTorch — callers don't need to check.

The package contains:

* :mod:`.compression` — the entry points (``tucker_reconstruct``,
  ``jl_project``, ``quantize_int8``). Each one tries the Triton kernel
  first and falls back to PyTorch if Triton is unavailable.
* :mod:`.tucker_reconstruct` — the actual Triton kernel for fused Tucker
  reconstruction. JIT-compiles on first call; subsequent calls reuse
  the cached binary.
"""

from kvcompress.kernels.triton.compression import is_triton_available

__all__ = ["is_triton_available"]
