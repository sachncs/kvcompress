"""Model-family adapters and the Hugging Face entry point.

This subpackage holds everything needed to integrate kvcompress with a
specific serving framework. The :mod:`.huggingface` module is the primary
integration (it patches Hugging Face's :class:`~transformers.cache_utils.DynamicCache`
so any HF causal LM gets compressed transparently). The
:mod:`.registry` module maps Hugging Face ``config.model_type`` strings
to family-specific shim modules. The per-family modules under
``adapters/<family>.py`` are no-ops today (the DynamicCache subclass
already covers them); they exist so the registry has a place to dispatch
to.

Future adapters (e.g. vLLM, TensorRT-LLM) will follow the same shape: a
top-level adapter module plus, when needed, family shims.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "HuggingFaceAdapter":
        from kvcompress.adapters.huggingface import HuggingFaceAdapter

        return HuggingFaceAdapter
    raise AttributeError(f"module 'kvcompress.adapters' has no attribute {name!r}")


__all__ = ["HuggingFaceAdapter"]
