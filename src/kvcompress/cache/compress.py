"""Compressed KV cache.

Stores compressed payloads per (layer, kind) and lazily reconstructs the
full K/V tensors on demand. Designed to be the lowest layer above raw
compressor payloads and below :class:`CacheManager`.

Public surface:

* :class:`CompressedKVCache` — per-call store/retrieve; thread-unsafe but
  re-entrant for single-inference workloads.
* :class:`CacheManager` — high-level facade for HF integration.
* :class:`LayerEntry` — one (layer, K, V) record held inside the cache.
* :func:`normalize_kv` — reshape HF-style K/V to ``(m, T, dh)``.
* :func:`payload_to_meta` — convert a payload to a
  :class:`~kvcompress.cache.metadata.LayerCompression`.

The split mirrors ``DynamicCache``'s split between raw storage and the
model-facing API.

Lifecycle:

1. Caller creates a :class:`CompressedKVCache` with a compressor and
   optional eviction cap.
2. ``store(layer, K, V)`` accepts K/V in either ``(B, n_kv, T, dh)`` or
   ``(n_kv, T, dh)`` layout (see :func:`normalize_kv`). It compresses
   each tensor via the compressor and stashes the
   :class:`~kvcompress.compressor.base.CompressedPayload`.
3. ``retrieve(layer)`` returns the reconstructed K/V pair. The cache
   itself never stores decompressed tensors — the compressor is called on
   every retrieve. Callers that need amortised cost should cache the
   result themselves.
4. Eviction is LRU when ``max_layers`` is set; otherwise the cache grows
   unbounded.

Thread-safety: not thread-safe. Single-inference workloads (one
generation loop) are fine; concurrent generations on the same cache
require external synchronisation.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

from kvcompress.cache.metadata import CompressionMetadata, LayerCompression
from kvcompress.compressor.base import CompressedPayload, KVCompressor

log = logging.getLogger(__name__)


@dataclass
class LayerEntry:
    """One (layer, K, V) record held inside :class:`CompressedKVCache`.

    Attributes:
        layer: layer index.
        key: compressed K payload, or ``None`` while a layer is being
            populated.
        value: compressed V payload, or ``None`` while a layer is being
            populated.
        extras: compressor-specific extra data (currently unused; kept
            for forward compatibility).
    """

    layer: int
    key: CompressedPayload | None = None
    value: CompressedPayload | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class CompressedKVCache:
    """In-memory compressed KV cache, layer-indexed.

    Args:
        compressor: compressor used to (de)compress every entry. Held by
            reference; the cache does not own its lifecycle.
        metadata: optional initial :class:`CompressionMetadata`. Mutated as
            entries are added.
        max_layers: optional cap on number of layers kept in memory; older
            layers are evicted LRU. ``None`` means no eviction.
        device: device tensors will be moved to on reconstruction. ``None``
            preserves the device of the stored factors.
    """

    def __init__(
        self,
        compressor: KVCompressor,
        metadata: CompressionMetadata | None = None,
        max_layers: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialise the cache.

        Args:
            compressor: compressor used to (de)compress every entry. Held
                by reference; the cache does not own its lifecycle.
            metadata: optional initial :class:`CompressionMetadata`.
                Mutated as entries are added. If ``None``, a fresh
                metadata object is created using ``compressor.name``.
            max_layers: optional cap on number of layers kept in memory;
                older layers are evicted LRU. ``None`` means no eviction.
            device: device tensors will be moved to on reconstruction.
                ``None`` preserves the device of the stored factors.
        """
        self.compressor = compressor
        self.metadata_ = metadata or CompressionMetadata(
            method=compressor.name,
            dtype="unknown",
        )
        self.max_layers = max_layers
        self.device = torch.device(device) if device is not None else None
        self.entries: OrderedDict[int, LayerEntry] = OrderedDict()

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def store(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        group_id: int = 0,
        group_size: int = 1,
        bits: tuple[int, ...] = (0, 2, 4, 8),
        seed: int = 0,
    ) -> None:
        """Compress and store the K/V pair for one layer.

        Args:
            layer: layer index.
            key: tensor of shape ``(B, n_kv, T, dh)`` or ``(n_kv, T, dh)``.
            value: same shape as ``key``.
            group_id: layer-group index used by the allocator (forwarded
                to ``CompressionMetadata`` so downstream tooling can split
                layers into groups).
            group_size: number of layers per group (forwarded to metadata).
            bits: residual bit-widths to consider.
            seed: per-layer seed.
        """
        k, v = normalize_kv(key, value)
        k_payload, v_payload = self.compressor.compress(k, v)
        entry = LayerEntry(layer=layer, key=k_payload, value=v_payload)
        self.entries[layer] = entry
        self.entries.move_to_end(layer)
        self.metadata_.add_layer(
            payload_to_meta(layer, "key", k, k_payload, seed, group_id, group_size)
        )
        self.metadata_.add_layer(
            payload_to_meta(layer, "value", v, v_payload, seed, group_id, group_size)
        )
        self.enforce_eviction()

    def retrieve(
        self,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct and return the K/V pair for ``layer``.

        Raises:
            KeyError: if ``layer`` isn't in the cache or isn't fully populated.
        """
        entry = self.entries.get(layer)
        if entry is None:
            raise KeyError(f"layer {layer} not in cache")
        if entry.key is None or entry.value is None:
            raise KeyError(f"layer {layer} is not fully populated")
        k, v = self.compressor.decompress(entry.key, entry.value)
        if self.device is not None:
            k = k.to(self.device)
            v = v.to(self.device)
        return k, v

    def payload(
        self,
        layer: int,
        kind: str,
    ) -> CompressedPayload:
        """Return the raw payload for one (layer, kind) cell.

        Raises:
            ValueError: if ``kind`` isn't ``"key"`` or ``"value"``.
            IndexError: if ``layer`` isn't in the cache.
        """
        entry = self.entries[layer]
        if kind == "key":
            return entry.key  # type: ignore[return-value]
        if kind == "value":
            return entry.value  # type: ignore[return-value]
        raise ValueError(f"kind must be 'key' or 'value', got {kind!r}")

    def has_layer(self, layer: int) -> bool:
        """Return ``True`` if the cache has an entry for ``layer``."""
        return layer in self.entries

    def clear(self) -> None:
        """Drop every entry and reset the metadata layer list."""
        self.entries.clear()
        self.metadata_.layers.clear()

    def evict_layer(self, layer: int) -> None:
        """Remove the entry for ``layer`` if present; no-op otherwise."""
        self.entries.pop(layer, None)
        self.metadata_.layers = [entry for entry in self.metadata_.layers if entry.layer != layer]

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    def memory_used(self) -> int:
        """Bytes occupied by all stored factors + residuals."""
        return sum(p.bytes_compressed for p in self.all_payloads())

    def memory_original(self) -> int:
        """Sum of ``bytes_original`` across all stored payloads."""
        return sum(p.bytes_original for p in self.all_payloads())

    def compression_ratio(self) -> float:
        """Achieved compression ratio across all stored payloads.

        Returns ``original / compressed``. Returns 1.0 if no payloads are
        stored (the "no cache" identity).
        """
        o = self.memory_original()
        c = self.memory_used()
        return o / c if c else 1.0

    def stats(self) -> dict[str, Any]:
        """Aggregate stats for logging and benchmarks.

        Returns:
            Dictionary with keys ``n_layers``, ``bytes_original``,
            ``bytes_compressed``, ``compression_ratio``, ``method``.
        """
        return {
            "n_layers": len(self.entries),
            "bytes_original": self.memory_original(),
            "bytes_compressed": self.memory_used(),
            "compression_ratio": self.compression_ratio(),
            "method": self.compressor.name,
        }

    def metadata(self) -> CompressionMetadata:
        """Return the live :class:`CompressionMetadata` (mutated in place)."""
        return self.metadata_

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def layers(self) -> Iterator[int]:
        """Iterate over live layer indices in insertion order."""
        return iter(list(self.entries.keys()))

    def __len__(self) -> int:
        """Number of layers currently stored."""
        return len(self.entries)

    def __contains__(self, layer: int) -> bool:
        """``layer in cache`` is ``True`` when the layer has an entry."""
        return layer in self.entries

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def all_payloads(self) -> list[CompressedPayload]:
        """Flatten K and V payloads across all live entries."""
        out: list[CompressedPayload] = []
        for e in self.entries.values():
            if e.key is not None:
                out.append(e.key)
            if e.value is not None:
                out.append(e.value)
        return out

    def enforce_eviction(self) -> None:
        """Drop the oldest entries until the cache fits under ``max_layers``.

        Uses LRU semantics (``OrderedDict.popitem(last=False)``) so the
        least-recently-stored layer is evicted first. The matching
        metadata entries are removed in the same pass.
        """
        if self.max_layers is None:
            return
        while len(self.entries) > self.max_layers:
            evicted_layer, _ = self.entries.popitem(last=False)
            log.debug("evicted layer %d (cache cap %d)", evicted_layer, self.max_layers)
            self.metadata_.layers = [
                entry for entry in self.metadata_.layers if entry.layer != evicted_layer
            ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_kv(
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape HF-style K/V to the ``(m, T, dh)`` layout the algorithm expects.

    Accepts ``(B, n_kv, T, dh)`` and ``(n_kv, T, dh)`` and returns
    ``(B·n_kv, T, dh)``. The reverse reshape on the way out restores the
    original layout.

    Why flatten the batch and head axes: the JoLT algorithm operates
    independently on each (B·n_kv) "merged head" slice, so treating the
    leading two axes as one keeps the algorithm simple.

    Raises:
        ValueError: if ``key`` and ``value`` have different shapes, or if
            ``key`` is not 3-D or 4-D.
    """
    if key.shape != value.shape:
        raise ValueError(f"K/V shape mismatch: {key.shape} vs {value.shape}")
    if key.dim() == 4:
        b, h, t, d = key.shape
        return key.reshape(b * h, t, d), value.reshape(b * h, t, d)
    if key.dim() == 3:
        return key, value
    raise ValueError(f"unsupported K/V rank: {key.dim()}")


def payload_to_meta(
    layer: int,
    kind: str,
    original: torch.Tensor,
    payload: CompressedPayload,
    seed: int,
    group_id: int = 0,
    group_size: int = 1,
) -> LayerCompression:
    """Convert a payload to a :class:`LayerCompression` metadata entry.

    Pulls the per-cell ``(r_token, r_feature, bits)`` out of the payload's
    ``metadata`` dict and stamps the original / compressed byte counts so
    the cache's stats are correct. The group_id and group_size are
    forwarded to metadata so downstream tooling (e.g. vLLM offload) can
    split layers into groups without re-deriving that information.

    Args:
        layer: layer index.
        kind: ``"key"`` or ``"value"``.
        original: the uncompressed K or V tensor.
        payload: the :class:`CompressedPayload` produced by the compressor.
        seed: seed used for JL / randomised SVD on this layer.
        group_id: layer-group index.
        group_size: number of layers per group.

    Returns:
        A :class:`LayerCompression` ready to be appended to
        ``CompressionMetadata.layers``.
    """
    return LayerCompression(
        layer=layer,
        kind=kind,
        m=payload.shape[0],
        tokens=payload.shape[1],
        dh=payload.shape[2],
        r_token=int(payload.metadata.get("r_token", 0)),
        r_feature=int(payload.metadata.get("r_feature", 0)),
        bits=int(payload.metadata.get("bits", 0)),
        core_dtype=str(payload.metadata.get("core_dtype", "fp16")),
        seed=seed,
        bytes_original=payload.bytes_original,
        bytes_compressed=payload.bytes_compressed,
        group_id=group_id,
        group_size=group_size,
    )


__all__ = ["CompressedKVCache", "LayerEntry", "normalize_kv", "payload_to_meta"]
