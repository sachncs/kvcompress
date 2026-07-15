"""Optional Triton kernels. Imported only when ``triton`` is installed."""

from kvcompress.kernels.triton.compression import is_triton_available

__all__ = ["is_triton_available"]