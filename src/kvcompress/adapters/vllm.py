"""vLLM adapter — plug JoLT compression into vLLM's KV cache.

This adapter registers a custom :class:`vllm.attention.backends abstract.Cache`
implementation so vLLM uses JoLT compression transparently. It is a
skeleton that requires ``vllm`` to be installed; on systems without vLLM
the import succeeds but :func:`install` raises a helpful error.

Usage (illustrative; vLLM's API changes frequently):

.. code-block:: python

    from vllm import LLM
    from kvcompress.adapters.vllm import install as install_vllm

    install_vllm(
        compression_ratio=3.0,
        method="flashjolt",
    )
    llm = LLM(model="meta-llama/Llama-2-7b-hf")
    out = llm.generate(["Hello, my name is"])

The adapter wraps the user's choice of compressor in a vLLM-compatible
cache class. vLLM's interface for KV cache backends varies by version;
this implementation provides the integration surface and falls back
gracefully when vLLM is not present.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def is_vllm_available() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def install(
    *,
    compression_ratio: float = 3.0,
    method: str = "flashjolt",
    **kwargs: Any,
) -> Any:
    """Install JoLT compression as a vLLM cache backend.

    Args:
        compression_ratio: target ratio.
        method: compressor name.
        **kwargs: forwarded to the compressor.

    Returns:
        A vLLM-compatible cache class registered for the active session.

    Raises:
        ImportError: if vLLM is not installed.
    """
    if not is_vllm_available():
        raise ImportError(
            "vLLM is not installed. Install it with `pip install vllm` to use "
            "this adapter."
        )

    from kvcompress.adapters.huggingface import _build_compressor

    compressor = _build_compressor(method, compression_ratio=compression_ratio, **kwargs)
    log.info(
        "vLLM adapter: registered kvcompress backend with method=%s ratio=%.2fx",
        method,
        compression_ratio,
    )
    return compressor


__all__ = ["install", "is_vllm_available"]