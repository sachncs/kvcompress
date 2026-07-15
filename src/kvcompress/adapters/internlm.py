"""InternLM family adapter.

InternLM uses standard GQA with rotary embeddings. DynamicCache
interception is sufficient.
"""

from __future__ import annotations

from typing import Any


def install(model: Any, cache_manager: Any) -> None:
    return None