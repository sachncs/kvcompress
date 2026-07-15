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

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.m, self.tokens, self.dh)

    @property
    def compression_ratio(self) -> float:
        if self.bytes_compressed == 0:
            return 1.0
        return self.bytes_original / self.bytes_compressed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LayerCompression":
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

    def layer(self, idx: int) -> LayerCompression:
        for entry in self.layers:
            if entry.layer == idx:
                return entry
        raise KeyError(f"no metadata for layer {idx}")

    def add_layer(self, entry: LayerCompression) -> None:
        for i, existing in enumerate(self.layers):
            if existing.layer == entry.layer and existing.kind == entry.kind:
                self.layers[i] = entry
                return
        self.layers.append(entry)

    def bytes_original(self) -> int:
        return sum(entry.bytes_original for entry in self.layers)

    def bytes_compressed(self) -> int:
        return sum(entry.bytes_compressed for entry in self.layers)

    def compression_ratio(self) -> float:
        o = self.bytes_original()
        c = self.bytes_compressed()
        return o / c if c else 1.0

    def to_dict(self) -> dict[str, Any]:
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
        return cls(
            method=d["method"],
            dtype=d["dtype"],
            layers=[LayerCompression.from_dict(x) for x in d.get("layers", [])],
            layer_groups=d.get("layer_groups", 1),
            bits_allowed=tuple(d.get("bits_allowed", (0, 2, 4, 8))),
            extras=d.get("extras", {}),
        )
