"""Compressed KV cache.

Stores compressed payloads per (layer, kind) and lazily reconstructs the
full K/V tensors on demand. Designed to be the lowest layer above raw
compressor payloads and below :class:`CacheManager`.

Public surface:

* :class:`CompressedKVCache` — per-call store/retrieve; thread-unsafe but
  re-entrant for single-inference workloads.
* :class:`CacheManager` — high-level facade for HF integration.

The split mirrors ``DynamicCache``'s split between raw storage and the
model-facing API.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

from kvcompress.cache.metadata import CompressionMetadata
from kvcompress.compressor.base import CompressedPayload, KVCompressor

log = logging.getLogger(__name__)


@dataclass
class _LayerEntry:
    """Internal per-(layer) entry.

    Holds the compressed key and value payloads plus optional precomputed
    JL projections for the residual path.
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
        self._compressor = compressor
        self._metadata = metadata or CompressionMetadata(
            method=compressor.name,
            dtype="unknown",
        )
        self._max_layers = max_layers
        self._device = torch.device(device) if device is not None else None
        self._entries: OrderedDict[int, _LayerEntry] = OrderedDict()

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
            group_id: layer-group index used by the allocator.
            group_size: number of layers per group.
            bits: residual bit-widths to consider.
            seed: per-layer seed.
        """
        k, v = _normalize_kv(key, value)
        k_payload, v_payload = self._compressor.compress(k, v)
        entry = _LayerEntry(layer=layer, key=k_payload, value=v_payload)
        self._entries[layer] = entry
        self._entries.move_to_end(layer)
        self._metadata.add_layer(_payload_to_meta(layer, "key", k, k_payload, seed))
        self._metadata.add_layer(_payload_to_meta(layer, "value", v, v_payload, seed))
        self._enforce_eviction()

    def retrieve(
        self,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct and return the K/V pair for one layer."""
        entry = self._entries.get(layer)
        if entry is None:
            raise KeyError(f"layer {layer} not in cache")
        if entry.key is None or entry.value is None:
            raise KeyError(f"layer {layer} is not fully populated")
        k, v = self._compressor.decompress(entry.key, entry.value)
        if self._device is not None:
            k = k.to(self._device)
            v = v.to(self._device)
        return k, v

    def payload(
        self,
        layer: int,
        kind: str,
    ) -> CompressedPayload:
        """Return the raw payload for one (layer, kind) cell."""
        entry = self._entries[layer]
        if kind == "key":
            return entry.key  # type: ignore[return-value]
        if kind == "value":
            return entry.value  # type: ignore[return-value]
        raise ValueError(f"kind must be 'key' or 'value', got {kind!r}")

    def has_layer(self, layer: int) -> bool:
        return layer in self._entries

    def clear(self) -> None:
        self._entries.clear()
        self._metadata.layers.clear()

    def evict_layer(self, layer: int) -> None:
        self._entries.pop(layer, None)
        self._metadata.layers = [
            l for l in self._metadata.layers if l.layer != layer
        ]

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    def memory_used(self) -> int:
        """Bytes occupied by all stored factors + residuals."""
        return sum(p.bytes_compressed for p in self._all_payloads())

    def memory_original(self) -> int:
        return sum(p.bytes_original for p in self._all_payloads())

    def compression_ratio(self) -> float:
        o = self.memory_original()
        c = self.memory_used()
        return o / c if c else 1.0

    def stats(self) -> dict[str, Any]:
        return {
            "n_layers": len(self._entries),
            "bytes_original": self.memory_original(),
            "bytes_compressed": self.memory_used(),
            "compression_ratio": self.compression_ratio(),
            "method": self._compressor.name,
        }

    def metadata(self) -> CompressionMetadata:
        return self._metadata

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def layers(self) -> Iterator[int]:
        return iter(list(self._entries.keys()))

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, layer: int) -> bool:
        return layer in self._entries

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _all_payloads(self) -> list[CompressedPayload]:
        out: list[CompressedPayload] = []
        for e in self._entries.values():
            if e.key is not None:
                out.append(e.key)
            if e.value is not None:
                out.append(e.value)
        return out

    def _enforce_eviction(self) -> None:
        if self._max_layers is None:
            return
        while len(self._entries) > self._max_layers:
            evicted_layer, evicted_entry = self._entries.popitem(last=False)
            log.debug("evicted layer %d (cache cap %d)", evicted_layer, self._max_layers)
            self._metadata.layers = [
                l for l in self._metadata.layers if l.layer != evicted_layer
            ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_kv(
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape HF-style K/V to the (m, T, dh) layout the algorithm expects.

    Accepts ``(B, n_kv, T, dh)`` and ``(n_kv, T, dh)`` and returns
    ``(B·n_kv, T, dh)``. The reverse reshape on the way out restores the
    original layout.
    """
    if key.shape != value.shape:
        raise ValueError(f"K/V shape mismatch: {key.shape} vs {value.shape}")
    if key.dim() == 4:
        b, h, t, d = key.shape
        return key.reshape(b * h, t, d), value.reshape(b * h, t, d)
    if key.dim() == 3:
        return key, value
    raise ValueError(f"unsupported K/V rank: {key.dim()}")


def _payload_to_meta(
    layer: int,
    kind: str,
    original: torch.Tensor,
    payload: CompressedPayload,
    seed: int,
) -> Any:
    from kvcompress.cache.metadata import LayerCompression

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
    )


__all__ = ["CompressedKVCache"]