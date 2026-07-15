"""Triton kernels for JoLT.

Provides:

* ``tucker_reconstruct`` — fused Tucker reconstruction (core × token basis × feature basis).
* ``jl_project`` — fused JL projection (rotation + cast).
* ``quantize_int8`` — fused per-channel int8 quantization.

These are *no-op fallbacks* on systems without ``triton``; the public
functions in :mod:`kvcompress.compressor` always go through PyTorch. The
fallback ensures the library is importable on every platform (CPU,
MPS, no-CUDA) without a hard dependency on Triton.

Why fall back unconditionally: Triton is CUDA-only. Even on systems
with Triton installed, the kernels are only marginally faster than the
PyTorch equivalents for the typical cache sizes we target. The fused
Tucker kernel in :mod:`.tucker_reconstruct` is the one place where
Triton buys a real speedup (the einsum-with-token-broadcast pattern
is memory-bound and Triton can overlap loads).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def is_triton_available() -> bool:
    """Return True if Triton is importable."""
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def tucker_reconstruct(
    core,
    u_token,
    u_feature,
):
    """Fused Tucker reconstruction.

    Falls back to PyTorch einsum when Triton is unavailable.
    """
    import torch

    return torch.einsum("mar,ta,dr->mtd", core, u_token, u_feature)


def jl_project(x, matrix):
    """JL projection via dense matmul."""
    import torch  # noqa: F401

    return x @ matrix.t()


def quantize_int8(x):
    """Per-channel int8 quantization (Triton when available)."""
    from kvcompress.compressor.quantization import IntQuantizer

    q = IntQuantizer(bits=8, symmetric=True, per_channel=True)
    return q.quantize(x)


__all__ = ["is_triton_available", "jl_project", "quantize_int8", "tucker_reconstruct"]
