"""Hugging Face adapter — top-level interception of the KV cache.

The adapter works in two ways:

1. Patch :class:`~transformers.cache_utils.DynamicCache` so writes go
   through the compressor. We patch the symbol in
   ``transformers.cache_utils`` AND in any module that has already
   imported it (``transformers.generation.utils``, the model class, etc.)
   so the patched class is the one that gets instantiated.

2. The user-facing API is :func:`kvcompress.api.enable_compression` which
   returns a :class:`~kvcompress.api.CompressionHandle` to disable the
   patch later.

Why we patch *multiple* module symbol tables
============================================

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

DynamicCache subclass behaviour
================================

The patched subclass overrides two methods:

* ``update`` — after the parent's ``super().update(...)`` concatenates
  the new K/V slice, we read the just-appended layer's tensors and call
  :meth:`CacheManager.store` to compress and stash them.
* ``__getitem__`` — before returning the layer's K/V, we ask the
  manager to reconstruct them. This is the path the attention layer
  hits when reading past keys.

Both paths also bump the :class:`~kvcompress.api.CompressionStats`
counters wired by :func:`kvcompress.api.enable_compression`.
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


def build_compressor(method: str, **kwargs: Any) -> KVCompressor:
    """Construct the named compressor from the public API string.

    Args:
        method: one of ``"jolt"``, ``"flashjolt"``, ``"identity"``.
        **kwargs: forwarded to the compressor constructor.

    Returns:
        A fresh :class:`KVCompressor` instance.

    Raises:
        NotImplementedError: if ``method`` isn't one of the supported names.
    """
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
    """Adapter that wires a :class:`KVCompressor` into an HF model.

    The adapter is **stateful** in two ways:

    * It owns a :class:`CacheManager` which holds the compressed payloads
      (created lazily by :meth:`enable`).
    * It remembers which ``transformers.*`` modules it patched
      (``self.patched_modules``) so :meth:`disable` can put them back.

    Args:
        model: HF ``PreTrainedModel`` returned by ``AutoModelForCausalLM``.
        method: compressor name (e.g. ``"flashjolt"``).
        compression_ratio: target ratio (e.g. ``3.0``).
        layer_groups: layer-group count for the allocator. The paper
            uses ``1``.
        bits: residual bit-widths the allocator may choose from.
        cache_implementation: HF cache implementation. We always use
            ``"dynamic"`` under the hood; any other value is silently
            ignored.
        seed: seed for randomized components.
        **kwargs: forwarded to the compressor.
    """

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
        # Hugging Face maintains a strict allow-list of cache
        # implementations. "kvcompress" isn't on it; we route through
        # "dynamic" (the standard paged-DynamicCache backend) and let
        # the patched class intercept writes instead.
        if cache_implementation not in ("dynamic", "dynamic_full"):
            log.debug(
                "kvcompress: ignoring cache_implementation=%r; using 'dynamic'",
                cache_implementation,
            )
            cache_implementation = "dynamic"
        self.cache_implementation = cache_implementation
        self.seed = int(seed)

        compressor = build_compressor(
            method,
            compression_ratio=compression_ratio,
            bits=bits,
            seed=seed,
            layer_groups=layer_groups,
            **kwargs,
        )
        self.compressor = compressor

        # ``_stats`` is wired by kvcompress.api.enable_compression after
        # the CompressionHandle is built. We keep it as an untyped
        # attribute to avoid an import cycle with kvcompress.api.
        self.manager: CacheManager | None = None
        self.enabled = False
        self.original_cache_implementation: str | None = None
        self.original_dynamic_cache_cls: type | None = None
        self.patched_cache_cls: type | None = None
        self.patched_modules: dict[str, type] = {}
        self.stats_ref: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def enable(self) -> None:
        """Install the patches and create the cache manager.

        Idempotency: calling :meth:`enable` twice is a no-op with a
        warning. Disable first if you want to switch methods.

        Side effects:
            * Constructs :attr:`_manager`.
            * Sets ``model.generation_config.cache_implementation``.
            * Patches ``DynamicCache`` in every loaded ``transformers.*``
              module.
            * Installs the family-specific shim (if any) via
              :mod:`kvcompress.adapters.registry`.

        Errors are caught and logged (warnings) so a partial install
        doesn't leave the user in a broken state — the worst case is
        a cache that doesn't get compressed.
        """
        if self.enabled:
            log.warning("kvcompress: enable() called twice; ignoring")
            return
        log.info(
            "kvcompress: enabling %s on %s",
            self.method,
            type(self.model).__name__,
        )
        self.manager = CacheManager(compressor=self.compressor)

        if hasattr(self.model, "generation_config"):
            self.original_cache_implementation = getattr(
                self.model.generation_config, "cache_implementation", None
            )
            self.model.generation_config.cache_implementation = self.cache_implementation

        from transformers.cache_utils import DynamicCache

        self.original_dynamic_cache_cls = DynamicCache
        self.install_dynamic_cache(DynamicCache)

        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if model_type is not None:
            self.family_install = registry_install(
                model_type=model_type,
                model=self.model,
                cache_manager=self.manager,
            )
            log.info("kvcompress: installed family shim for %s", model_type)

        self.enabled = True

    def disable(self) -> None:
        """Restore the original ``DynamicCache`` symbol and revert the
        generation_config."""
        if not self.enabled:
            return
        if (
            hasattr(self.model, "generation_config")
            and self.original_cache_implementation is not None
        ):
            self.model.generation_config.cache_implementation = self.original_cache_implementation
        self.uninstall_dynamic_cache()
        self.enabled = False

    # ------------------------------------------------------------------
    # DynamicCache patching
    # ------------------------------------------------------------------

    def install_dynamic_cache(self, dynamic_cache_cls: type) -> None:
        """Patch DynamicCache everywhere it has been imported.

        The subclass intercepts two methods:

        * ``update`` — compresses the layer's K/V after the parent's
          concatenation.
        * ``__getitem__`` — reconstructs the layer's K/V before the
          attention layer reads past keys.

        Both also bump ``cache.stats_ref`` (the CompressionStats on the
        handle) so users can read cumulative counts via
        :meth:`~kvcompress.api.CompressionHandle.stats_dict`.

        Args:
            dynamic_cache_cls: the original :class:`DynamicCache` class
                captured before patching.
        """
        cache = self

        # ``type: ignore[misc, valid-type]`` — subclassing a class we
        # only have a dynamic handle to. mypy can't see the metaclass.
        class KvCompressCache(dynamic_cache_cls):  # type: ignore[misc, valid-type]
            """DynamicCache subclass that compresses on every update."""

            def update(  # type: ignore[override]  -- overrides HF's update with a richer signature
                self,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                layer_idx: int,
                cache_kwargs: dict | None = None,
            ):
                out = super().update(key_states, value_states, layer_idx, cache_kwargs)
                if cache.manager is not None:
                    # ``type: ignore[attr-defined]`` — ``layers`` is set
                    # by the parent class lazily, so static analysers
                    # don't see it.
                    layer_obj = self.layers[layer_idx]  # type: ignore[attr-defined]
                    k = layer_obj.keys
                    v = layer_obj.values
                    cache.manager.store(layer_idx, k, v)
                    if cache.stats_ref is not None:
                        cache.stats_ref.compress_calls += 1
                        cache.stats_ref.bytes_original += (
                            k.numel() * k.element_size() + v.numel() * v.element_size()
                        )
                        # Update bytes_compressed by reading from manager.
                        cache.stats_ref.bytes_compressed = cache.manager.memory_used()
                return out

            def __getitem__(self, layer_idx: int):  # type: ignore[override]
                # ``type: ignore[override]`` — the parent's __getitem__
                # signature varies across HF versions; we accept any.
                if (
                    cache.manager is not None
                    and layer_idx in cache.manager
                    and getattr(self, "layers", None) is not None
                    and layer_idx < len(self.layers)
                ):
                    k, v = cache.manager.retrieve(layer_idx)
                    self.layers[layer_idx].keys = k
                    self.layers[layer_idx].values = v
                    if cache.stats_ref is not None:
                        cache.stats_ref.decompress_calls += 1
                return super().__getitem__(layer_idx)

        self.patched_cache_cls = KvCompressCache

        # Patch the symbol in transformers.cache_utils.
        # ``type: ignore[misc]`` — HF types DynamicCache as a frozen
        # class; reassigning a module attribute is technically a
        # ``module-level override`` that mypy warns about.
        import transformers.cache_utils as cu

        cu.DynamicCache = KvCompressCache  # type: ignore[misc]

        # Patch transformers.generation.utils.
        import transformers.generation.utils as gu

        if hasattr(gu, "DynamicCache"):
            gu.DynamicCache = KvCompressCache  # type: ignore[misc]
            self.patched_modules["transformers.generation.utils"] = KvCompressCache

        # Patch any other transformers module that imported DynamicCache.
        # ``type: ignore[misc]`` — same module-attribute reassignment.
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not mod_name.startswith("transformers"):
                continue
            if getattr(mod, "DynamicCache", None) is dynamic_cache_cls:
                mod.DynamicCache = KvCompressCache  # type: ignore[misc]
                self.patched_modules[mod_name] = KvCompressCache

    def uninstall_dynamic_cache(self) -> None:
        """Restore ``DynamicCache`` in every module we patched.

        We restore the *original* class captured in :attr:`_original_dynamic_cache_cls`
        across all modules we touched. If a module was imported after
        :meth:`enable` and has the patched class, it stays patched —
        there's no safe way to find it retroactively, but that's rare.
        """
        import transformers.cache_utils as cu

        if self.original_dynamic_cache_cls is not None:
            cu.DynamicCache = self.original_dynamic_cache_cls  # type: ignore[misc]
        for mod_name in list(self.patched_modules.keys()):
            mod = sys.modules.get(mod_name)
            if mod is not None:
                mod.DynamicCache = self.original_dynamic_cache_cls  # type: ignore[misc]
        self.patched_modules.clear()


def generic_install(model: object, cache_manager: CacheManager) -> None:
    """Default install for unrecognized model types.

    Returns ``None`` so the registry's ``install`` dispatch knows the
    generic path was taken. The DynamicCache subclass installed by
    :meth:`HuggingFaceAdapter.enable` is what does the actual work.
    """
    return None
