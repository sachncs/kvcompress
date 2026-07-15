"""Mistral family adapter.

Mistral uses sliding-window attention with grouped-query attention. The
default DynamicCache subclass handles standard K/V; the sliding-window
behaviour is managed by HF's cache layer, which we do not interfere with.
"""

from __future__ import annotations

from typing import Any


def install(model: Any, cache_manager: Any) -> None:
    return None