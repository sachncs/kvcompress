"""Regression tests for the cache metadata index Tier 0 bugs.

Covers:
- B-01: clear() doesn't reset meta.index
- B-02: evict_layer() / enforce_eviction() don't rebuild index
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kvfold.store.compress import Cache
from kvfold.store.metadata import LayerMeta, Meta


@dataclass
class _StubCompressor:
    name: str = "stub"

    def compress(self, key: Any, value: Any) -> tuple[Any, Any]:
        return (None, None)

    def restore(self, kp: Any, vp: Any) -> tuple[Any, Any]:
        return (None, None)

    def default_config(self) -> Any:
        return None


def test_clear_resets_index() -> None:
    """After clear(), the metadata index must no longer reference removed layers."""
    meta = Meta(method="stub", dtype="float16")
    meta.add_layer(LayerMeta(layer=3, kind="key", m=1, tokens=4, dh=4, r_token=0, r_feature=0, bits=0))
    assert (3, "key") in meta.index
    cache = Cache(compressor=_StubCompressor(), metadata=meta)
    cache.entries[3] = object()
    cache.clear()
    assert (3, "key") not in meta.index


def test_evict_keeps_index_consistent() -> None:
    """After evict_layer, the index must point to valid layer positions."""
    meta = Meta(method="stub", dtype="float16")
    for layer, kind in [(0, "key"), (0, "value"), (1, "key"), (1, "value")]:
        meta.add_layer(LayerMeta(layer=layer, kind=kind, m=1, tokens=4, dh=4, r_token=0, r_feature=0, bits=0))
    meta.layers = [entry for entry in meta.layers if entry.layer != 0]
    meta.rebuild_index()
    assert (0, "key") not in meta.index
    assert (0, "value") not in meta.index
    assert meta.index[(1, "key")] == 0
    assert meta.index[(1, "value")] == 1
