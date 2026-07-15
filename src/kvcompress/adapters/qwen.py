"""Qwen family adapter (Qwen2 and Qwen2-MoE).

Qwen2 uses GQA with rotary embeddings. DynamicCache subclass interception
is sufficient; no model-specific patches are required.
"""

from __future__ import annotations

from typing import Any


def install(model: Any, cache_manager: Any) -> None:
    return None