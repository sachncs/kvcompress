"""Hugging Face adapter — top-level interception of the KV cache.

The adapter wraps the model's ``_prepare_cache_for_generation`` hook so only
that model's cache creation is scoped to the compressor. The user-facing API
is :func:`kvfold.api.enable_compression`, which
   returns a :class:`~kvfold.api.CompressionHandle` to disable the
   patch later.

DynamicCache subclass behaviour
================================

The patched subclass owns only metadata and compressed payloads. It does not
call the parent's ``update`` implementation because that implementation would
retain complete K/V tensors in ``DynamicLayer.keys`` and ``DynamicLayer.values``.
Updated tensors are reconstructed and concatenated only for the current
attention operation, then the persistent cache is compressed immediately.

Both paths also bump the :class:`~kvfold.api.CompressionStats`
counters wired by :func:`kvfold.api.enable_compression`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import threading
import weakref
from typing import Any

import torch

from kvfold.adapter.base import Adapter, AdapterState
from kvfold.api import build_compressor
from kvfold.store.manager import Pool

__all__ = ["HF", "is_compression_active"]


log = logging.getLogger(__name__)

# Weak-keyed registry of installed adapters. Model-scoped hooks may coexist.
# ``weakref.WeakValueDictionary`` lets GC reclaim adapters when the model
# goes out of scope (e.g. between test cases) without us having to thread
# teardown through every call site.
INSTALLED_ADAPTERS: weakref.WeakKeyDictionary[object, HF] = weakref.WeakKeyDictionary()
_LIFECYCLE_LOCK = threading.RLock()


def is_compression_active(model: object) -> bool:
    """Return ``True`` if a :class:`HF` is active on ``model``."""
    return model in INSTALLED_ADAPTERS


def _resolve_dynamic_cache() -> type:
    """Resolve and validate the Transformers cache contract before mutation."""
    from kvfold.errors import TransformersCompatibilityError

    try:
        from transformers import cache_utils
    except ImportError as exc:
        raise TransformersCompatibilityError(
            "Transformers cache utilities are unavailable",
            hint="install a supported transformers>=4.45,<5 release",
        ) from exc
    dynamic_cache = getattr(cache_utils, "DynamicCache", None)
    if not isinstance(dynamic_cache, type):
        raise TransformersCompatibilityError(
            "Transformers does not expose a DynamicCache class",
            hint="install a supported transformers>=4.45,<5 release",
        )
    missing = [
        name
        for name in ("update", "__getitem__")
        if not callable(getattr(dynamic_cache, name, None))
    ]
    if missing:
        raise TransformersCompatibilityError(
            f"Transformers DynamicCache is missing required methods: {', '.join(missing)}",
            hint="upgrade or pin transformers to a supported release",
        )
    return dynamic_cache


class HF(Adapter[AdapterState]):
    """Adapter that wires a :class:`Compressor` into an HF model.

    The adapter is **stateful** in two ways:

    * It owns a :class:`Pool` which holds the compressed payloads
      (created lazily by :meth:`enable`).
    * It remembers the model-scoped cache hook so :meth:`disable` can restore
      it exactly.

    Args:
        model: HF ``PreTrainedModel`` returned by ``AutoModelForCausalLM``.
        method: compressor name (e.g. ``"flash"``).
        ratio: target ratio (e.g. ``3.0``).
        layer_groups: layer-group count for the allocator. The paper
            uses ``1``.
        bits: residual bit-widths the allocator may choose from.
        cache_implementation: ``"dynamic"`` or ``"dynamic_full"``.
            Both use the dynamic cache subclass under the hood; other values
            fail early.
        seed: seed for randomized components.
        device: device tensors should live on. ``None`` infers from the
            model's first parameter.
        **kwargs: forwarded to the compressor.
    """

    def __init__(
        self,
        *,
        model: Any,
        method: str,
        ratio: float,
        layer_groups: int = 1,
        bits: tuple[int, ...] = (0, 2, 4, 8),
        cache_implementation: str = "dynamic",
        seed: int = 0,
        device: torch.device | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self.method = method
        self.ratio = float(ratio)
        self.layer_groups = int(layer_groups)
        self.bits = tuple(bits)
        if cache_implementation not in ("dynamic", "dynamic_full"):
            raise ValueError(
                "cache_implementation must be 'dynamic' or 'dynamic_full'; "
                f"got {cache_implementation!r}"
            )
        self.cache_implementation = cache_implementation
        self.seed = int(seed)

        # Resolve device: explicit > model parameter > cpu.
        if device is None:
            try:
                param = next(model.parameters(), None)
                if param is not None:
                    device = param.device
            except (StopIteration, AttributeError):
                device = "cpu"
        self.device = torch.device(device) if isinstance(device, str) else device

        # Build compressor kwargs: pass-through methods don't accept bits/seed;
        # only forward kwargs the chosen method's config actually accepts.
        from kvfold.config import REGISTRY as CONFIG_REGISTRY

        config_cls = None
        try:
            config_cls = CONFIG_REGISTRY.resolve(method)
            valid_fields = {f.name for f in dataclasses.fields(config_cls)}
        except KeyError:
            valid_fields = set()
        unknown = set(kwargs) - valid_fields
        if unknown:
            from kvfold.errors import MethodConfigError

            raise MethodConfigError(
                method,
                ",".join(sorted(unknown)),
                f"unknown field(s); valid: {sorted(valid_fields)}",
            )
        compressor_kwargs: dict[str, Any] = dict(kwargs)
        if "ratio" in valid_fields:
            compressor_kwargs["ratio"] = ratio
        bits_field = getattr(config_cls, "__dataclass_fields__", {}).get("bits")
        # Allocator methods accept candidate bit-width tuples; fixed-width
        # methods (int2/int4/int8) deliberately expose a scalar config and
        # must keep their own validated default.
        if (
            "bits" in valid_fields
            and bits_field is not None
            and isinstance(bits_field.default, tuple)
        ):
            compressor_kwargs["bits"] = bits
        if "seed" in valid_fields:
            compressor_kwargs["seed"] = seed
        compressor = build_compressor(method, **compressor_kwargs)
        self.compressor = compressor

        # ``stats`` is wired by kvfold.api.enable_compression after
        # the CompressionHandle is built. We keep it as an untyped
        # attribute to avoid an import cycle with kvfold.api.
        self.manager: Pool | None = None
        self.enabled = False
        # Snapshot of state we mutated during enable(); populated as we
        # mutate so :meth:`disable` can restore verbatim. Cleared on
        # successful disable.
        self.original_cache_implementation: str | None = None
        self.cache_implementation_existed: bool = False
        self.original_prepare_cache: Any = None
        self.prepare_cache_wrapper: Any = None
        from kvfold.api import CompressionStats

        self.stats_ref: CompressionStats | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self) -> AdapterState:
        """Attach the Hugging Face cache adapter and return its state."""
        self.enable()
        return self.state()

    def detach(self) -> None:
        """Detach the adapter and restore the framework's original symbols."""
        self.disable()

    def state(self) -> AdapterState:
        """Return an immutable lifecycle snapshot."""
        return AdapterState(name="huggingface", attached=self.enabled)

    def enable(self) -> None:
        """Install the patches and create the cache manager.

        Idempotency: a second call raises :class:`RuntimeError` instead
        of silently no-oping (matches the docstring contract in
        ``api.py``).

        Errors are caught and rolled back: every side effect is recorded
        so :meth:`disable` can put the runtime back to its pre-enable
        state, even if enable() blew up halfway through.
        """
        with _LIFECYCLE_LOCK:
            self._enable_locked()

    def _enable_locked(self) -> None:
        """Perform enable while holding the process-wide lifecycle lock."""
        if self.enabled:
            raise RuntimeError(
                "kvfold: enable() called twice on the same adapter; call handle.disable() first."
            )
        if is_compression_active(self.model):
            raise RuntimeError(
                "kvfold: a different adapter is already installed on this model; "
                "call its handle.disable() first."
            )
        log.info(
            "kvfold: enabling %s on %s",
            self.method,
            type(self.model).__name__,
        )
        try:
            self.enable_inner()
        except Exception:
            # Roll back any side effects already applied.
            self.rollback_partial_enable()
            raise

    def enable_inner(self) -> None:
        """Inner enable; assumes caller handles rollback on failure."""
        dynamic_cache_cls = _resolve_dynamic_cache()
        self.manager = Pool(
            compressor=self.compressor,
            device=self.device,
        )

        if hasattr(self.model, "generation_config"):
            self.cache_implementation_existed = hasattr(
                self.model.generation_config, "cache_implementation"
            )
            self.original_cache_implementation = getattr(
                self.model.generation_config, "cache_implementation", None
            )
            self.model.generation_config.cache_implementation = self.cache_implementation

        self.install_dynamic_cache(dynamic_cache_cls)

        self.enabled = True
        self.attached = True
        INSTALLED_ADAPTERS[self.model] = self

    def rollback_partial_enable(self) -> None:
        """Undo whatever side effects :meth:`enable` completed before failing."""
        # Undo the model-scoped hook if it was installed.
        try:
            self.uninstall_dynamic_cache()
        except (AttributeError, TypeError, RuntimeError):
            log.exception("kvfold: rollback failed during uninstall_dynamic_cache")
        # Undo generation_config mutation.
        if hasattr(self.model, "generation_config"):
            try:
                if self.cache_implementation_existed:
                    self.model.generation_config.cache_implementation = (
                        self.original_cache_implementation
                    )
                else:
                    with contextlib.suppress(AttributeError):
                        delattr(self.model.generation_config, "cache_implementation")
            except (AttributeError, TypeError, RuntimeError):
                log.exception("kvfold: rollback failed for cache_implementation")
        self.manager = None
        self.attached = False

    def disable(self) -> None:
        """Restore the original cache symbols and generation configuration.

        The operation is idempotent and serialized with other lifecycle
        transitions.
        """
        with _LIFECYCLE_LOCK:
            self._disable_locked()

    def _disable_locked(self) -> None:
        """Perform disable while holding the process-wide lifecycle lock."""
        if not self.enabled:
            return
        if hasattr(self.model, "generation_config") and self.cache_implementation_existed:
            self.model.generation_config.cache_implementation = self.original_cache_implementation
        elif hasattr(self.model, "generation_config") and hasattr(
            self.model.generation_config, "cache_implementation"
        ):
            # Generation config had no cache_implementation before enable;
            # remove the attribute we added so we don't leak a side effect.
            with contextlib.suppress(AttributeError):
                delattr(self.model.generation_config, "cache_implementation")
        self.uninstall_dynamic_cache()
        self.enabled = False
        self.attached = False
        INSTALLED_ADAPTERS.pop(self.model, None)
        log.info("kvfold: disabled compression on %s", type(self.model).__name__)

    # ------------------------------------------------------------------
    # DynamicCache patching
    # ------------------------------------------------------------------

    def install_dynamic_cache(self, dynamic_cache_cls: type) -> None:
        """Install the compressed DynamicCache through the model hook.

        The subclass intercepts two methods:

        * ``update`` — reconstructs the prior compressed state, appends the
          new slice, and stores only the compressed result.
        * ``__getitem__`` — reconstructs the layer's K/V for callers using
          the legacy cache indexing protocol.

        Both also bump ``cache.stats_ref`` (the CompressionStats on the
        handle) so users can read cumulative counts via
        :meth:`~kvfold.api.CompressionHandle.stats_dict`.

        Args:
            dynamic_cache_cls: the original :class:`DynamicCache` class
                captured before patching.
        """
        cache = self

        class KvCompressCache(dynamic_cache_cls):
            """DynamicCache-compatible cache with compressed backing storage."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._kvfold_lengths: dict[int, int] = {}
                self._kvfold_cumulative_lengths: dict[int, int] = {}
                self._kvfold_shapes: dict[int, tuple[int, ...]] = {}
                # Keep layer objects as framework metadata, but never let them
                # own the persistent K/V tensors.
                for layer in self.layers:
                    layer.keys = None
                    layer.values = None
                    layer.is_initialized = False

            def _ensure_layer(self, layer_idx: int) -> Any:
                if layer_idx < len(self.layers):
                    return self.layers[layer_idx]
                if self.layer_class_to_replicate is None:
                    raise IndexError(f"cache has no layer {layer_idx}")
                while len(self.layers) <= layer_idx:
                    self.layers.append(self.layer_class_to_replicate())
                return self.layers[layer_idx]

            def _stored_states(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
                if cache.manager is None or layer_idx not in cache.manager:
                    return None
                states = cache.manager.retrieve(layer_idx)
                shape = self._kvfold_shapes.get(layer_idx)
                if shape is not None and states[0].dim() != len(shape):
                    states = (
                        states[0].reshape(shape),
                        states[1].reshape(shape),
                    )
                if cache.stats_ref is not None:
                    cache.stats_ref.record_decompress()
                return states

            def _store_states(
                self,
                layer_idx: int,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
            ) -> None:
                if cache.manager is None:
                    return
                cache.manager.store(layer_idx, key_states, value_states)
                if cache.stats_ref is not None:
                    cache.stats_ref.record_compress(
                        bytes_original=cache.manager.memory_original(),
                        bytes_compressed=cache.manager.memory_used(),
                        bytes_resident=cache.manager.memory_resident(),
                    )

            def update(
                self,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                layer_idx: int,
                cache_kwargs: dict | None = None,
            ):
                """Append K/V while keeping only compressed persistent state.

                Args:
                    key_states: K tensor for this layer's new tokens.
                    value_states: V tensor for this layer's new tokens.
                    layer_idx: index of the layer being updated.
                    cache_kwargs: HF's optional cache kwargs (e.g.
                        ``sin/cos`` for RoPE models).

                Returns:
                    Whatever the HF parent returns (a tuple of updated
                    K, V that callers may use directly).
                """
                layer = self._ensure_layer(layer_idx)
                if cache.manager is None:
                    return super().update(key_states, value_states, layer_idx, cache_kwargs)

                previous = self._stored_states(layer_idx)
                if previous is None:
                    full_keys, full_values = key_states, value_states
                    cumulative_length = key_states.shape[-2]
                else:
                    old_keys, old_values = previous
                    full_keys = torch.cat((old_keys, key_states), dim=-2)
                    full_values = torch.cat((old_values, value_states), dim=-2)
                    cumulative_length = (
                        self._kvfold_cumulative_lengths[layer_idx] + key_states.shape[-2]
                    )

                # Match DynamicSlidingWindowLayer: return the current window,
                # but persist only window-1 states for the next update.
                if getattr(layer, "is_sliding", False):
                    window = int(layer.sliding_window)
                    stored_keys = full_keys[..., -window + 1 :, :]
                    stored_values = full_values[..., -window + 1 :, :]
                else:
                    stored_keys, stored_values = full_keys, full_values

                self._store_states(layer_idx, stored_keys, stored_values)
                self._kvfold_lengths[layer_idx] = stored_keys.shape[-2]
                self._kvfold_cumulative_lengths[layer_idx] = cumulative_length
                self._kvfold_shapes[layer_idx] = tuple(stored_keys.shape)
                # Never assign full tensors to layer.keys/values.
                return full_keys, full_values

            def __getitem__(self, layer_idx: int):
                states = self._stored_states(layer_idx)
                if states is None:
                    raise KeyError(f"cache has no compressed layer {layer_idx}")
                return states

            def __iter__(self):
                for layer_idx in range(len(self)):
                    yield self[layer_idx]

            def __len__(self) -> int:
                return len(self._kvfold_lengths)

            def get_seq_length(self, layer_idx: int = 0) -> int:
                return self._kvfold_cumulative_lengths.get(layer_idx, 0)

            def get_max_cache_shape(self, layer_idx: int = 0) -> int:
                if layer_idx < len(self.layers) and getattr(
                    self.layers[layer_idx], "is_sliding", False
                ):
                    return int(self.layers[layer_idx].sliding_window)
                return -1

            def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int):
                query_length = cache_position.shape[0]
                layer = self._ensure_layer(layer_idx)
                if getattr(layer, "is_sliding", False):
                    window = int(layer.sliding_window)
                    cumulative = self.get_seq_length(layer_idx)
                    return (
                        min(cumulative, window - 1) + query_length,
                        max(cumulative - window + 1, 0),
                    )
                return self.get_seq_length(layer_idx) + query_length, 0

            def to_legacy_cache(self):
                return tuple(self[layer_idx] for layer_idx in range(len(self)))

            def reset(self) -> None:
                if cache.manager is not None:
                    cache.manager.clear()
                self._kvfold_lengths.clear()
                self._kvfold_cumulative_lengths.clear()
                self._kvfold_shapes.clear()

            def crop(self, max_length: int) -> None:
                for layer_idx in list(self._kvfold_lengths):
                    keys, values = self[layer_idx]
                    limit = max_length
                    if limit < 0:
                        limit = self.get_seq_length(layer_idx) - abs(limit)
                    if keys.shape[-2] > limit:
                        self._store_states(layer_idx, keys[..., :limit, :], values[..., :limit, :])
                        self._kvfold_lengths[layer_idx] = limit
                        self._kvfold_cumulative_lengths[layer_idx] = limit

            def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
                for layer_idx in list(self._kvfold_lengths):
                    keys, values = self[layer_idx]
                    selected_keys = keys.index_select(0, beam_idx.to(keys.device))
                    selected_values = values.index_select(0, beam_idx.to(values.device))
                    self._store_states(
                        layer_idx,
                        selected_keys,
                        selected_values,
                    )
                    self._kvfold_shapes[layer_idx] = tuple(selected_keys.shape)

            def batch_repeat_interleave(self, repeats: int) -> None:
                for layer_idx in list(self._kvfold_lengths):
                    keys, values = self[layer_idx]
                    repeated_keys = keys.repeat_interleave(repeats, dim=0)
                    repeated_values = values.repeat_interleave(repeats, dim=0)
                    self._store_states(
                        layer_idx,
                        repeated_keys,
                        repeated_values,
                    )
                    self._kvfold_shapes[layer_idx] = tuple(repeated_keys.shape)

            def batch_select_indices(self, indices: torch.Tensor) -> None:
                for layer_idx in list(self._kvfold_lengths):
                    keys, values = self[layer_idx]
                    selected_keys = keys[indices, ...]
                    selected_values = values[indices, ...]
                    self._store_states(layer_idx, selected_keys, selected_values)
                    self._kvfold_shapes[layer_idx] = tuple(selected_keys.shape)

        if not self.install_model_cache_hook(KvCompressCache):
            from kvfold.errors import TransformersCompatibilityError

            raise TransformersCompatibilityError(
                "model does not expose _prepare_cache_for_generation",
                hint="use a supported transformers>=4.45,<5 model implementation",
            )

    def install_model_cache_hook(self, cache_cls: type) -> bool:
        """Install a model-scoped DynamicCache factory when supported.

        Returns ``True`` when the model exposes the supported Transformers
        generation hook. The generation module is patched only for the
        duration of this model's cache construction.
        """
        original = getattr(self.model, "_prepare_cache_for_generation", None)
        if not callable(original):
            return False
        import transformers.generation.utils as gu

        def prepare_cache(*args: Any, **kwargs: Any) -> Any:
            with _LIFECYCLE_LOCK:
                previous = getattr(gu, "DynamicCache", None)
                gu.__dict__["DynamicCache"] = cache_cls
                try:
                    return original(*args, **kwargs)
                finally:
                    if getattr(gu, "DynamicCache", None) is cache_cls:
                        gu.__dict__["DynamicCache"] = previous

        self.original_prepare_cache = original
        self.prepare_cache_wrapper = prepare_cache
        self.model.__dict__["_prepare_cache_for_generation"] = prepare_cache
        log.debug("kvfold: using model-scoped Transformers cache hook")
        return True

    def uninstall_dynamic_cache(self) -> None:
        """Restore the model-scoped cache preparation hook."""
        if self.prepare_cache_wrapper is not None:
            current = getattr(self.model, "_prepare_cache_for_generation", None)
            if current is self.prepare_cache_wrapper:
                self.model.__dict__["_prepare_cache_for_generation"] = self.original_prepare_cache
            self.prepare_cache_wrapper = None
            self.original_prepare_cache = None
