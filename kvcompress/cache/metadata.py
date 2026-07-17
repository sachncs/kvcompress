"""Compression metadata — the serializable side of a compressed cache.

Every layer of a compressed cache carries a small dataclass describing its
layout. The payload's tensor factors live separately; only this dataclass
needs to be JSON-serializable for safetensors round-trips.

Two dataclasses:

* :class:`LayerCompression` — one cell (layer × kind × K/V). Carries the
  ``r_token``, ``r_feature``, ``bits`` chosen by the allocator plus the
  bytes-original / bytes-compressed bookkeeping that
  :class:`CompressedKVCache` uses to report memory.
* :class:`CompressionMetadata` — top-level metadata for one cache. Holds a
  list of :class:`LayerCompression` entries plus the cache-wide settings
  (method, dtype, layer-group count, allowed bit-widths, calibration
  extras).

Both classes implement :meth:`to_dict` / :meth:`from_dict` for round-trip
serialisation. The dict shape is stable across versions: any field that
exists today stays, new fields may be appended.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class LayerCompression:
    """Metadata for one (layer, key-or-value) compressed tensor.

    Attributes:
        layer: layer index in the model.
        kind: ``"key"`` or ``"value"``.
        m: product of head count and number of layers in the group (``|g|·n_h``).
        tokens: number of tokens cached so far.
        dh: per-head feature dim.
        r_token: token-mode rank.
        r_feature: feature-mode rank.
        bits: residual bit-width (0, 2, 4, or 8).
        core_dtype: dtype of the Tucker core (``fp16`` or ``fp32``).
        seed: seed used for JL / randomized SVD on this layer.
        bytes_original: size of the original (m, T, dh) tensor in bytes.
        bytes_compressed: bytes occupied by the stored factors + residual.
        group_id: layer-group index (forwarded by the caller; downstream
            tooling like vLLM offload uses it to bucket layers).
        group_size: number of layers per group (forwarded by the caller).
    """

    layer: int
    kind: str
    m: int
    tokens: int
    dh: int
    r_token: int
    r_feature: int
    bits: int
    core_dtype: str = "fp16"
    seed: int = 0
    bytes_original: int = 0
    bytes_compressed: int = 0
    group_id: int = 0
    group_size: int = 1

    @property
    def shape(self) -> tuple[int, int, int]:
        """Original (m, tokens, dh) shape of the cell."""
        return (self.m, self.tokens, self.dh)

    @property
    def compression_ratio(self) -> float:
        """Achieved compression ratio (original / compressed). 1.0 if empty."""
        if self.bytes_compressed == 0:
            return 1.0
        return self.bytes_original / self.bytes_compressed

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-friendly)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LayerCompression":
        """Inverse of :meth:`to_dict`."""
        return cls(**d)


@dataclass
class CompressionMetadata:
    """Top-level metadata for one :class:`CompressedKVCache`.

    Attributes:
        method: compressor name.
        dtype: original tensor dtype (e.g. ``torch.bfloat16``).
        layers: per-layer entries.
        layer_groups: number of layer groups used by the allocator.
        bits_allowed: tuple of residual bit-widths the allocator chose from.
        extras: compressor-specific serializable extras (e.g. JL projection
            shapes, calibration constants).
    """

    method: str
    dtype: str
    layers: list[LayerCompression] = field(default_factory=list)
    layer_groups: int = 1
    bits_allowed: tuple[int, ...] = (0, 2, 4, 8)
    extras: dict[str, Any] = field(default_factory=dict)
    # Ponytail: O(1) (layer, kind) -> index lookup alongside the list.
    # The list stays the source of truth for ordering; the dict shadows
    # it for membership checks. ``_rebuild_index`` resyncs after bulk
    # operations that mutate the list directly (tests, deserialise).
    _index: dict[tuple[int, str], int] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def layer(self, idx: int) -> LayerCompression:
        """Return the first :class:`LayerCompression` for ``idx``.

        Raises:
            KeyError: if no entry matches.
        """
        for entry in self.layers:
            if entry.layer == idx:
                return entry
        raise KeyError(f"no metadata for layer {idx}")

    def add_layer(self, entry: LayerCompression) -> None:
        """Insert ``entry``, replacing any existing (layer, kind) match.

        Replacing rather than appending means a re-store of the same
        (layer, kind) pair updates the metadata in place rather than
        accumulating duplicates.
        """
        key = (entry.layer, entry.kind)
        if key in self._index:
            self.layers[self._index[key]] = entry
            return
        self._index[key] = len(self.layers)
        self.layers.append(entry)

    def _rebuild_index(self) -> None:
        """Resync the ``(layer, kind) -> index`` dict from ``layers``.

        Call this after bulk-mutating ``self.layers`` directly (e.g. in
        ``__post_init__`` after deserialisation).
        """
        self._index = {(entry.layer, entry.kind): i for i, entry in enumerate(self.layers)}

    def bytes_original(self) -> int:
        """Sum of uncompressed bytes across all layer entries."""
        return sum(entry.bytes_original for entry in self.layers)

    def bytes_compressed(self) -> int:
        """Sum of compressed bytes across all layer entries."""
        return sum(entry.bytes_compressed for entry in self.layers)

    def compression_ratio(self) -> float:
        """Aggregate compression ratio across all entries."""
        o = self.bytes_original()
        c = self.bytes_compressed()
        return o / c if c else 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "method": self.method,
            "dtype": self.dtype,
            "layers": [entry.to_dict() for entry in self.layers],
            "layer_groups": self.layer_groups,
            "bits_allowed": list(self.bits_allowed),
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompressionMetadata":
        """Inverse of :meth:`to_dict`."""
        meta = cls(
            method=d["method"],
            dtype=d["dtype"],
            layers=[LayerCompression.from_dict(x) for x in d.get("layers", [])],
            layer_groups=d.get("layer_groups", 1),
            bits_allowed=tuple(d.get("bits_allowed", (0, 2, 4, 8))),
            extras=d.get("extras", {}),
        )
        meta._rebuild_index()
        return meta
