"""Hugging Face adapter — top-level interception of the KV cache.

The adapter works in two ways:

1. Patch ``DynamicCache`` so writes go through the compressor. We patch
   the symbol in ``transformers.cache_utils`` AND in any module that has
   already imported it (``transformers.generation.utils``, the model
   class, etc.) so the patched class is the one that gets instantiated.

2. The user-facing API is :func:`kvcompress.api.enable_compression` which
   returns a :class:`~kvcompress.api.CompressionHandle` to disable the
   patch later.

Why we patch *multiple* module symbol tables:

Hugging Face's :class:`~transformers.cache_utils.DynamicCache` is
imported by name in many places:
``transformers.cache_utils.DynamicCache``,
``transformers.generation.utils.DynamicCache``, and per-model
``DynamicCache`` imports. After ``import transformers.cache_utils as cu;
cu.DynamicCache = X``, the module attribute on ``cache_utils`` is
``X`` but a *previously-imported* name in ``generation.utils`` is
still the original class. Methods that look up ``DynamicCache`` by
name in their enclosing namespace see the original. To override that
we walk all loaded ``transformers.*`` modules and reassign their
``DynamicCache`` attribute.

This pattern is the same one used by ``accelerate`` for device
placement and by some HF callback libraries for tracing.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import torch

from kvcompress.adapters.registry import install as registry_install
from kvcompress.cache.manager import CacheManager
from kvcompress.compressor.base import KVCompressor
from kvcompress.compressor.flashjolt import FlashJoLTCompressor
from kvcompress.compressor.jolt import JoLTCompressor

log = logging.getLogger(__name__)


def _build_compressor(method: str, **kwargs: Any) -> KVCompressor:
    method = method.lower()
    if method == "jolt":
        return JoLTCompressor(**kwargs)
    if method == "flashjolt":
        return FlashJoLTCompressor(**kwargs)
    if method == "identity":
        from kvcompress.compressor.identity import IdentityCompressor

        return IdentityCompressor(**kwargs)
    raise NotImplementedError(
        f"compressor method {method!r} is not implemented in this milestone; "
        "supported methods so far: 'jolt', 'flashjolt', 'identity'."
    )


class HuggingFaceAdapter:
    """Adapter that wires a :class:`KVCompressor` into an HF model."""

    def __init__(
        self,
        *,
        model: Any,
        method: str,
        compression_ratio: float,
        layer_groups: int = 1,
        bits: tuple[int, ...] = (0, 2, 4, 8),
        cache_implementation: str = "dynamic",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.method = method
        self.compression_ratio = float(compression_ratio)
        self.layer_groups = int(layer_groups)
        self.bits = tuple(bits)
        # We always use 'dynamic' under the hood.
        if cache_implementation not in ("dynamic", "dynamic_full"):
            log.debug(
                "kvcompress: ignoring cache_implementation=%r; using 'dynamic'",
                cache_implementation,
            )
            cache_implementation = "dynamic"
        self.cache_implementation = cache_implementation
        self.seed = int(seed)

        compressor = _build_compressor(
            method,
            compression_ratio=compression_ratio,
            bits=bits,
            seed=seed,
            layer_groups=layer_groups,
            **kwargs,
        )
        self.compressor = compressor

        self._manager: CacheManager | None = None
        self._enabled = False
        self._original_cache_implementation: str | None = None
        self._original_dynamic_cache_cls: type | None = None
        self._patched_cache_cls: type | None = None
        self._patched_modules: dict[str, type] = {}
        self._stats = None  # set by api.enable_compression

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def enable(self) -> None:
        if self._enabled:
            log.warning("kvcompress: enable() called twice; ignoring")
            return
        log.info(
            "kvcompress: enabling %s on %s",
            self.method,
            type(self.model).__name__,
        )
        self._manager = CacheManager(compressor=self.compressor)

        if hasattr(self.model, "generation_config"):
            self._original_cache_implementation = getattr(
                self.model.generation_config, "cache_implementation", None
            )
            self.model.generation_config.cache_implementation = self.cache_implementation

        try:
            from transformers.cache_utils import DynamicCache

            self._original_dynamic_cache_cls = DynamicCache
            self._install_dynamic_cache(DynamicCache)
        except Exception as e:  # pragma: no cover
            log.warning("kvcompress: failed to patch DynamicCache: %s", e)

        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if model_type is not None:
            try:
                self._family_install = registry_install(
                    model_type=model_type,
                    model=self.model,
                    cache_manager=self._manager,
                )
                log.info("kvcompress: installed family shim for %s", model_type)
            except Exception as e:  # pragma: no cover
                log.warning("kvcompress: family install failed: %s", e)

        self._enabled = True

    def disable(self) -> None:
        if not self._enabled:
            return
        if (
            hasattr(self.model, "generation_config")
            and self._original_cache_implementation is not None
        ):
            self.model.generation_config.cache_implementation = self._original_cache_implementation
        try:
            self._uninstall_dynamic_cache()
        except Exception:  # pragma: no cover
            pass
        self._enabled = False

    # ------------------------------------------------------------------
    # DynamicCache patching
    # ------------------------------------------------------------------

    def _install_dynamic_cache(self, dynamic_cache_cls: type) -> None:
        """Patch DynamicCache everywhere it has been imported."""
        cache = self

        class _KvCompressCache(dynamic_cache_cls):  # type: ignore[misc, valid-type]
            """DynamicCache subclass that compresses on every update."""

            def update(  # type: ignore[override]
                self,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                layer_idx: int,
                cache_kwargs: dict | None = None,
            ):
                out = super().update(key_states, value_states, layer_idx, cache_kwargs)
                if cache._manager is not None:
                    layer_obj = self.layers[layer_idx]  # type: ignore[attr-defined]
                    k = layer_obj.keys
                    v = layer_obj.values
                    cache._manager.store(layer_idx, k, v)
                    if cache._stats is not None:
                        cache._stats.compress_calls += 1
                        cache._stats.bytes_original += (
                            k.numel() * k.element_size() + v.numel() * v.element_size()
                        )
                        # Update bytes_compressed by reading from manager.
                        cache._stats.bytes_compressed = cache._manager.memory_used()
                return out

            def __getitem__(self, layer_idx: int):  # type: ignore[override]
                if (
                    cache._manager is not None
                    and layer_idx in cache._manager
                    and getattr(self, "layers", None) is not None
                    and layer_idx < len(self.layers)
                ):
                    k, v = cache._manager.retrieve(layer_idx)
                    self.layers[layer_idx].keys = k
                    self.layers[layer_idx].values = v
                    if cache._stats is not None:
                        cache._stats.decompress_calls += 1
                return super().__getitem__(layer_idx)

        self._patched_cache_cls = _KvCompressCache

        # Patch the symbol in transformers.cache_utils.
        import transformers.cache_utils as cu

        cu.DynamicCache = _KvCompressCache  # type: ignore[misc]

        # Patch transformers.generation.utils.
        try:
            import transformers.generation.utils as gu

            if hasattr(gu, "DynamicCache"):
                gu.DynamicCache = _KvCompressCache  # type: ignore[misc]
                self._patched_modules["transformers.generation.utils"] = _KvCompressCache
        except Exception:  # pragma: no cover
            pass

        # Patch any other transformers module that imported DynamicCache.
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not mod_name.startswith("transformers"):
                continue
            try:
                if getattr(mod, "DynamicCache", None) is dynamic_cache_cls:
                    mod.DynamicCache = _KvCompressCache  # type: ignore[misc]
                    self._patched_modules[mod_name] = _KvCompressCache
            except Exception:  # pragma: no cover
                pass

    def _uninstall_dynamic_cache(self) -> None:
        import transformers.cache_utils as cu

        if self._original_dynamic_cache_cls is not None:
            cu.DynamicCache = self._original_dynamic_cache_cls  # type: ignore[misc]
        for mod_name in list(self._patched_modules.keys()):
            mod = sys.modules.get(mod_name)
            if mod is not None:
                try:
                    mod.DynamicCache = self._original_dynamic_cache_cls  # type: ignore[misc]
                except Exception:  # pragma: no cover
                    pass
        self._patched_modules.clear()


def _generic_install(model: object, cache_manager: CacheManager) -> None:
    return None
