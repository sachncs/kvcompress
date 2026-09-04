"""Model family registry.

Maps ``config.model_type`` (as exposed by Hugging Face ``transformers``) to
the appropriate shim module that knows how to wire :class:`Compressor`
into that family's attention layer.

Adding a new family: write ``adapters/<name>.py`` exposing
``install(model, cache_manager)`` and add an entry to :data:`REGISTRY`.

Today every entry is a no-op shim because the :class:`HF`'s
:class:`~transformers.cache_utils.DynamicCache` subclass already covers the
standard cache layout. The registry exists so future model-specific hooks
(custom attention kernels, MLA, fused QKV) have a place to land.

Thread-safety: the registry is mutated only at import time and via
:func:`register`. The module uses a module-level dict without locking;
callers that register at runtime must do so before any
:class:`HF` is constructed.
"""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)

# Registry of supported model types. Keys are HF ``config.model_type``
# strings; values are dotted module paths to the family shim.
REGISTRY: dict[str, str] = {
    "llama": "kvfold.adapter.llama",
    "mistral": "kvfold.adapter.mistral",
    "qwen2": "kvfold.adapter.qwen",
    "qwen2_moe": "kvfold.adapter.qwen",
    "gemma": "kvfold.adapter.gemma",
    "gemma2": "kvfold.adapter.gemma",
    "phi": "kvfold.adapter.phi",
    "phi3": "kvfold.adapter.phi",
    "mixtral": "kvfold.adapter.mixtral",
    "falcon": "kvfold.adapter.falcon",
    "deepseek": "kvfold.adapter.deepseek",
    "internlm": "kvfold.adapter.internlm",
}


def known_model_types() -> list[str]:
    """Sorted list of all registered ``model_type`` strings."""
    return sorted(REGISTRY.keys())


def resolve(model_type: str) -> str | None:
    """Return the dotted module path that handles ``model_type``, or None."""
    return REGISTRY.get(model_type)


def register(model_type: str, module_path: str) -> None:
    """Register a custom family shim.

    Args:
        model_type: the HF ``config.model_type`` string to dispatch.
        module_path: dotted path to a module exposing ``install(model, cache_manager)``.

    Raises:
        ValueError: if ``model_type`` is already registered.
    """
    if model_type in REGISTRY:
        raise ValueError(f"model_type {model_type!r} already registered")
    REGISTRY[model_type] = module_path


def install(model: object, cache_manager: object, model_type: str) -> Callable[[], None] | None:
    """Dispatch to the right family shim and invoke its ``install``.

    If ``model_type`` isn't registered, falls through to the generic
    path (``generic_install``) which is a no-op — the DynamicCache
    subclass does the real work.

    Args:
        model: the HF model being patched.
        cache_manager: the :class:`Pool` to pass to the shim.
        model_type: HF ``config.model_type``.

    Returns:
        The shim's ``install`` callable (if a shim was used), or ``None``
        for the generic path.
    """
    module_path = resolve(model_type)
    if module_path is None:
        log.warning(
            "kvfold: no shim for model_type=%s; using generic interception",
            model_type,
        )
        from kvfold.adapter.huggingface import generic_install

        generic_install(model, cache_manager)
        return None
    import importlib

    module = importlib.import_module(module_path)
    return module.install(model, cache_manager)
