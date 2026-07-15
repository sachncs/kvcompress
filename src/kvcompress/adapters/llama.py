"""Llama family adapter.

Llama uses ``LlamaAttention`` with rotary embeddings and standard
``past_key_values``. The DynamicCache subclass installed by
:class:`HuggingFaceAdapter` already intercepts cache writes correctly;
this shim exists as a documented no-op so the registry can dispatch to
a family-specific module.

For RoPE pre/post handling on keys, see :class:`kvcompress.compressor.jolt`
which compresses pre-RoPE by default.
"""

from __future__ import annotations

from typing import Any


def install(model: Any, cache_manager: Any) -> None:
    """No-op for Llama; the generic DynamicCache subclass is sufficient."""
    return None