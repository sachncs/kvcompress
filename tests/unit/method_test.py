from __future__ import annotations

import pytest

from kvfold.method import family, is_dtype_only, is_low, is_passthrough, is_quant, normalise


@pytest.mark.parametrize(
    ("name", "expected"),
    [("JOLT", "jolt"), (" flash ", "flash"), ("INT4", "int4")],
)
def test_normalise_accepts_supported_names(name: str, expected: str) -> None:
    assert normalise(name) == expected


def test_method_helpers_classify_families() -> None:
    assert family("jolt") == "low"
    assert family("int4") == "quant"
    assert family("fp16") == "dtype"
    assert family("pass") == "pass"
    assert is_low("flash")
    assert is_quant("int8")
    assert is_dtype_only("bf16")
    assert is_passthrough("pass")


def test_normalise_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        normalise("does-not-exist")
