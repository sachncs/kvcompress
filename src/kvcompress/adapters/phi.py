"""Phi family adapter (Phi-1/2 and Phi-3).

Phi uses partial rotary embeddings and fused QKV projections.
DynamicCache interception is sufficient.
"""

from __future__ import annotations

from typing import Any


def install(model: Any, cache_manager: Any) -> None:
    return None