"""Falcon family adapter.

Falcon uses multi-query attention (MQA) and fused QKV; DynamicCache
interception is sufficient.
"""

from __future__ import annotations

from typing import Any


def install(model: Any, cache_manager: Any) -> None:
    return None