"""Model family registry.

Maps ``config.model_type`` (as exposed by Hugging Face ``transformers``) to
the appropriate shim module that knows how to wire :class:`KVCompressor`
into that family's attention layer.

Adding a new family: write ``adapters/<name>.py`` exposing
``install(model, cache_manager)`` and add an entry to :data:`_REGISTRY`.

Today every entry is a no-op shim because the :class:`HuggingFaceAdapter`'s
:class:`~transformers.cache_utils.DynamicCache` subclass already covers the
standard cache layout. The registry exists so future model-specific hooks
(custom attention kernels, MLA, fused QKV) have a place to land.
"""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)

# Registry of supported model types.
_REGISTRY: dict[str, str] = {
    "llama": "kvcompress.adapters.llama",
    "mistral": "kvcompress.adapters.mistral",
    "qwen2": "kvcompress.adapters.qwen",
    "qwen2_moe": "kvcompress.adapters.qwen",
    "gemma": "kvcompress.adapters.gemma",
    "gemma2": "kvcompress.adapters.gemma",
    "phi": "kvcompress.adapters.phi",
    "phi3": "kvcompress.adapters.phi",
    "mixtral": "kvcompress.adapters.mixtral",
    "falcon": "kvcompress.adapters.falcon",
    "deepseek": "kvcompress.adapters.deepseek",
    "internlm": "kvcompress.adapters.internlm",
}


def known_model_types() -> list[str]:
    return sorted(_REGISTRY.keys())


def resolve(model_type: str) -> str | None:
    """Return the module path that handles ``model_type``, or None."""
    return _REGISTRY.get(model_type)


def register(model_type: str, module_path: str) -> None:
    """Register a custom adapter. Raises if already registered."""
    if model_type in _REGISTRY:
        raise ValueError(f"model_type {model_type!r} already registered")
    _REGISTRY[model_type] = module_path


def install(model: object, cache_manager: object, model_type: str) -> Callable[[], None] | None:
    """Dispatch to the right family shim.

    Returns the shim's ``install`` callable so the caller can invoke it.
    """
    module_path = resolve(model_type)
    if module_path is None:
        log.warning(
            "kvcompress: no shim for model_type=%s; using generic interception",
            model_type,
        )
        from kvcompress.adapters.huggingface import _generic_install

        _generic_install(model, cache_manager)
        return None
    import importlib

    module = importlib.import_module(module_path)
    return module.install(model, cache_manager)
