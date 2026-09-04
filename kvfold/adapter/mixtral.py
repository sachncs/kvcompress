"""Mixtral (Mixture of Experts) adapter.

Mixtral uses sparse MoE FFN with sliding-window GQA attention. The KV
cache layout is the same as Mistral; we apply no model-specific patch.
"""

from __future__ import annotations

from typing import Any


def install(model: Any, cache_manager: Any) -> None:
    """No-op for Mixtral.

    The sparse experts only affect the FFN; the attention KV cache uses
    the same layout as Mistral. ``model_type`` is ``"mixtral"`` for the
    MoE checkpoint and ``"qwen2_moe"`` for the analogous Qwen MoE, both
    routed here.
    """
    return None
