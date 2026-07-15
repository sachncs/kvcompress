"""Tests for the cache metadata round-trip."""

from __future__ import annotations

import pytest

from kvcompress.cache.metadata import (
    CompressionMetadata,
    LayerCompression,
)


def test_layer_compression_to_from_dict_roundtrip() -> None:
    lc = LayerCompression(
        layer=5,
        kind="key",
        m=8,
        tokens=64,
        dh=32,
        r_token=16,
        r_feature=8,
        bits=4,
        core_dtype="fp16",
        seed=42,
        bytes_original=65536,
        bytes_compressed=8192,
        group_id=0,
        group_size=1,
    )
    d = lc.to_dict()
    lc2 = LayerCompression.from_dict(d)
    assert lc2.layer == lc.layer
    assert lc2.kind == lc.kind
    assert lc2.m == lc.m
    assert lc2.tokens == lc.tokens
    assert lc2.dh == lc.dh
    assert lc2.r_token == lc.r_token
    assert lc2.r_feature == lc.r_feature
    assert lc2.bits == lc.bits
    assert lc2.core_dtype == lc.core_dtype
    assert lc2.seed == lc.seed
    assert lc2.bytes_original == lc.bytes_original
    assert lc2.bytes_compressed == lc.bytes_compressed
    assert lc2.group_id == lc.group_id
    assert lc2.group_size == lc.group_size


def test_layer_compression_shape_property() -> None:
    lc = LayerCompression(
        layer=0, kind="key", m=4, tokens=128, dh=64, r_token=8, r_feature=4, bits=0
    )
    assert lc.shape == (4, 128, 64)


def test_layer_compression_compression_ratio_property() -> None:
    lc = LayerCompression(
        layer=0,
        kind="key",
        m=4,
        tokens=128,
        dh=64,
        r_token=8,
        r_feature=4,
        bits=0,
        bytes_original=1000,
        bytes_compressed=500,
    )
    assert lc.compression_ratio == 2.0
    # No payload → 1.0 (the no-cache identity).
    lc2 = LayerCompression(
        layer=0,
        kind="key",
        m=4,
        tokens=128,
        dh=64,
        r_token=8,
        r_feature=4,
        bits=0,
        bytes_compressed=0,
    )
    assert lc2.compression_ratio == 1.0


def test_compression_metadata_layer_lookup() -> None:
    meta = CompressionMetadata(method="jolt", dtype="float16")
    meta.add_layer(
        LayerCompression(layer=0, kind="key", m=4, tokens=64, dh=32, r_token=8, r_feature=4, bits=0)
    )
    meta.add_layer(
        LayerCompression(
            layer=0, kind="value", m=4, tokens=64, dh=32, r_token=8, r_feature=4, bits=0
        )
    )
    meta.add_layer(
        LayerCompression(layer=1, kind="key", m=4, tokens=64, dh=32, r_token=8, r_feature=4, bits=0)
    )
    assert meta.layer(0).kind == "key"
    assert meta.layer(0).m == 4
    assert meta.layer(1).kind == "key"
    with pytest.raises(KeyError, match="no metadata for layer 99"):
        meta.layer(99)


def test_compression_metadata_add_layer_replaces() -> None:
    """Adding a layer entry with the same (layer, kind) replaces."""
    meta = CompressionMetadata(method="jolt", dtype="float16")
    a = LayerCompression(
        layer=0,
        kind="key",
        m=4,
        tokens=64,
        dh=32,
        r_token=4,
        r_feature=4,
        bits=0,
        bytes_compressed=100,
    )
    b = LayerCompression(
        layer=0,
        kind="key",
        m=4,
        tokens=64,
        dh=32,
        r_token=4,
        r_feature=4,
        bits=0,
        bytes_compressed=200,
    )
    meta.add_layer(a)
    meta.add_layer(b)
    # Only one entry for (layer=0, kind=key), with the second's bytes.
    assert len(meta.layers) == 1
    assert meta.layer(0).bytes_compressed == 200


def test_compression_metadata_bytes_aggregates() -> None:
    meta = CompressionMetadata(method="jolt", dtype="float16")
    meta.add_layer(
        LayerCompression(
            layer=0,
            kind="key",
            m=4,
            tokens=64,
            dh=32,
            r_token=8,
            r_feature=4,
            bits=0,
            bytes_original=1000,
            bytes_compressed=300,
        )
    )
    meta.add_layer(
        LayerCompression(
            layer=0,
            kind="value",
            m=4,
            tokens=64,
            dh=32,
            r_token=8,
            r_feature=4,
            bits=0,
            bytes_original=1000,
            bytes_compressed=500,
        )
    )
    assert meta.bytes_original() == 2000
    assert meta.bytes_compressed() == 800
    assert meta.compression_ratio() == 2.5


def test_compression_metadata_to_from_dict_roundtrip() -> None:
    meta = CompressionMetadata(
        method="flashjolt",
        dtype="bfloat16",
        layer_groups=4,
        bits_allowed=(0, 2, 4, 8),
        extras={"calibration": "default"},
    )
    meta.add_layer(
        LayerCompression(
            layer=3,
            kind="key",
            m=8,
            tokens=256,
            dh=64,
            r_token=16,
            r_feature=8,
            bits=4,
            core_dtype="fp16",
            seed=7,
            bytes_original=4096,
            bytes_compressed=1024,
        )
    )
    d = meta.to_dict()
    meta2 = CompressionMetadata.from_dict(d)
    assert meta2.method == meta.method
    assert meta2.dtype == meta.dtype
    assert meta2.layer_groups == meta.layer_groups
    assert meta2.bits_allowed == meta.bits_allowed
    assert meta2.extras == meta.extras
    assert len(meta2.layers) == 1
    assert meta2.layer(3).r_token == 16
    assert meta2.layer(3).bits == 4


def test_compression_metadata_empty_layer() -> None:
    meta = CompressionMetadata(method="jolt", dtype="float16")
    assert meta.layers == []
    assert meta.bytes_original() == 0
    assert meta.bytes_compressed() == 0
    assert meta.compression_ratio() == 1.0


def test_compression_metadata_default_bits_allowed() -> None:
    """Default ``bits_allowed`` is the paper's grid ``(0, 2, 4, 8)``."""
    meta = CompressionMetadata(method="jolt", dtype="float16")
    assert meta.bits_allowed == (0, 2, 4, 8)
