"""Mixtral (Mixture of Experts) adapter.

Mixtral uses sparse MoE FFN with sliding-window GQA attention. The KV
cache layout is the same as Mistral; we apply no model-specific patch.
"""

from __future__ import annotations

from typing import Any


def install(model: Any, cache_manager: Any) -> None:
    return None