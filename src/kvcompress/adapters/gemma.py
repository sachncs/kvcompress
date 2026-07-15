"""Gemma family adapter (Gemma and Gemma2).

Gemma models use post-norm transformers with RoPE. Standard
DynamicCache interception is sufficient.
"""

from __future__ import annotations

from typing import Any


def install(model: Any, cache_manager: Any) -> None:
    return None