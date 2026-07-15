"""Falcon family adapter.

Falcon uses multi-query attention (MQA) and fused QKV; DynamicCache
interception is sufficient.
"""

from __future__ import annotations

from typing import Any


def install(model: Any, cache_manager: Any) -> None:
    """No-op for Falcon.

    Falcon's MQA means n_kv=1 — the head axis is a singleton. Partial
    Tucker still works (the algorithm doesn't care), but the speedup
    vs. plain low-rank is smaller because there's less redundancy to
    absorb across the head dimension.
    """
    return None
