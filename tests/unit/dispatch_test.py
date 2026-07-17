"""Tests for :mod:`kvcompress.compressor.dispatch`.

The dispatcher is the public surface that maps the ``method`` string from
:func:`kvcompress.api.enable_compression` to a concrete compressor. Bugs
here manifest as silently broken ``enable_compression(method="int4")``
calls, so we cover the happy path for every method plus the failure
modes that previously surfaced only at runtime.
"""

from __future__ import annotations

import pytest
import torch

from kvcompress import build_compressor, supported_methods
from kvcompress.compressor.dispatch import METHODS


# ---------------------------------------------------------------------------
# Happy-path: every advertised method returns the right concrete class.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "expected_class_name"),
    [
        ("jolt", "JoLTCompressor"),
        ("flashjolt", "FlashJoLTCompressor"),
        ("lowrank", "LowRankCompressor"),
        ("int2", "IntQuantOnlyCompressor"),
        ("int4", "IntQuantOnlyCompressor"),
        ("int8", "IntQuantOnlyCompressor"),
        ("fp8", "IdentityCompressor"),
        ("fp16", "IdentityCompressor"),
        ("bf16", "IdentityCompressor"),
        ("identity", "IdentityCompressor"),
    ],
)
def test_dispatch_returns_correct_class(method: str, expected_class_name: str) -> None:
    c = build_compressor(method)
    assert type(c).__name__ == expected_class_name


def test_supported_methods_is_complete() -> None:
    """Every entry in :data:`METHODS` is reachable from ``supported_methods``."""
    methods = supported_methods()
    assert set(methods) == set(METHODS.keys())
    # README documents 9 methods (jolt/flashjolt/lowrank/int2/int4/int8/fp8/fp16/identity).
    # We added bf16 on top; ensure at least the documented set is present.
    required = {"jolt", "flashjolt", "lowrank", "int2", "int4", "int8", "fp8", "fp16", "identity"}
    assert required.issubset(set(methods))


# ---------------------------------------------------------------------------
# Argument plumbing
# ---------------------------------------------------------------------------


def test_int_methods_default_bits_to_method_name() -> None:
    """``int4`` should set ``bits=4`` even if the caller didn't pass it."""
    c = build_compressor("int4")
    assert c.bits == 4


def test_int_methods_allow_caller_bits_override() -> None:
    """Caller-provided ``bits`` wins over the per-method default."""
    c = build_compressor("int4", bits=8)
    assert c.bits == 8


def test_lowrank_forwards_rank() -> None:
    c = build_compressor("lowrank", rank=128)
    assert c.rank == 128


def test_fp16_forces_dtype() -> None:
    c = build_compressor("fp16")
    assert c.factor_dtype == torch.float16


def test_bf16_forces_dtype() -> None:
    c = build_compressor("bf16")
    assert c.factor_dtype == torch.bfloat16


def test_jolt_forwards_compression_ratio() -> None:
    c = build_compressor("jolt", compression_ratio=4.0)
    assert c.compression_ratio == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Failure modes: unknown method, unknown kwargs
# ---------------------------------------------------------------------------


def test_unknown_method_raises_with_actionable_message() -> None:
    with pytest.raises(NotImplementedError, match="not supported"):
        build_compressor("not-a-method")


def test_unknown_kwarg_raises_with_actionable_message() -> None:
    """A typo (``per_chanel`` instead of ``per_channel``) must fail loud."""
    with pytest.raises(ValueError, match="unexpected kwargs"):
        build_compressor("int4", per_chanel=True)


def test_int_kwargs_rejected_on_identity() -> None:
    """``per_channel`` is meaningless for IdentityCompressor — reject."""
    with pytest.raises(ValueError, match="unexpected kwargs"):
        build_compressor("identity", per_channel=True)


def test_method_is_case_insensitive() -> None:
    """``"INT4"`` and ``"Int4"`` should both work — we lowercase internally."""
    a = build_compressor("INT4")
    b = build_compressor("Int4")
    assert type(a) is type(b)


# ---------------------------------------------------------------------------
# Round-trip smoke test through dispatch (not just construction)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["jolt", "flashjolt", "lowrank", "int2", "int4", "int8"])
def test_round_trip_on_tiny_tensor(method: str) -> None:
    """Each method's ``compress``/``decompress`` returns the right shape.

    We don't check reconstruction fidelity here — that lives in the
    compressor-specific test files. This test only catches "the dispatch
    wired up the wrong class" or "decompress returned a different shape".
    """
    torch.manual_seed(0)
    k = torch.randn(2, 8, 4)
    v = torch.randn(2, 8, 4)
    if method in ("jolt", "flashjolt"):
        c = build_compressor(method, compression_ratio=2.0)
    elif method == "lowrank":
        c = build_compressor(method, rank=4)
    else:
        c = build_compressor(method)
    kp, vp = c.compress(k, v)
    k_hat, v_hat = c.decompress(kp, vp)
    assert k_hat.shape == k.shape
    assert v_hat.shape == v.shape
    assert k_hat.dtype == k.dtype
    assert v_hat.dtype == v.dtype
