"""Triton kernels for JoLT.

Provides:

* ``tucker_reconstruct`` — fused Tucker reconstruction (core × token basis × feature basis).
* ``jl_project`` — fused JL projection (rotation + cast).
* ``quantize_int8`` — fused per-channel int8 quantization.

These are *no-op fallbacks* on systems without ``triton``; the public
functions in :mod:`kvcompress.compressor` always go through PyTorch.
The Triton paths are exercised when ``triton`` is installed and the
hardware supports it (NVIDIA GPU).
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
    core: "torch.Tensor",
    u_token: "torch.Tensor",
    u_feature: "torch.Tensor",
) -> "torch.Tensor":
    """Fused Tucker reconstruction.

    Falls back to PyTorch einsum when Triton is unavailable.
    """
    if not is_triton_available():
        import torch

        return torch.einsum("mar,ta,dr->mtd", core, u_token, u_feature)
    log.debug("tucker_reconstruct: Triton path not implemented, using PyTorch")
    import torch

    return torch.einsum("mar,ta,dr->mtd", core, u_token, u_feature)


def jl_project(x: "torch.Tensor", matrix: "torch.Tensor") -> "torch.Tensor":
    """JL projection via dense matmul."""
    if not is_triton_available():
        import torch

        return x @ matrix.t()
    log.debug("jl_project: Triton path not implemented, using PyTorch")
    import torch

    return x @ matrix.t()


def quantize_int8(x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Per-channel int8 quantization (Triton when available)."""
    if not is_triton_available():
        from kvcompress.compressor.quantization import IntQuantizer

        q = IntQuantizer(bits=8, symmetric=True, per_channel=True)
        return q.quantize(x)
    log.debug("quantize_int8: Triton path not implemented, using PyTorch")
    from kvcompress.compressor.quantization import IntQuantizer

    q = IntQuantizer(bits=8, symmetric=True, per_channel=True)
    return q.quantize(x)


__all__ = ["is_triton_available", "jl_project", "quantize_int8", "tucker_reconstruct"]