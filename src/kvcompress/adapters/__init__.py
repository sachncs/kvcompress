"""Hugging Face adapter — top-level import shim.

The real implementation lives in :mod:`kvcompress.adapters.huggingface`.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "HuggingFaceAdapter":
        from kvcompress.adapters.huggingface import HuggingFaceAdapter

        return HuggingFaceAdapter
    raise AttributeError(f"module 'kvcompress.adapters' has no attribute {name!r}")


__all__ = ["HuggingFaceAdapter"]