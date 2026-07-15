"""DeepSeek family adapter.

DeepSeek-V2 and V3 use MLA (multi-head latent attention). The cache layout
differs from standard MHA/GQA: K and V are projected to a single latent
vector per head. Our compressor operates on the materialized K/V
returned by the model, so DynamicCache interception is sufficient.
"""

from __future__ import annotations

from typing import Any


def install(model: Any, cache_manager: Any) -> None:
    return None