"""Tests for the kvfold error hierarchy."""

from __future__ import annotations

import pytest

from kvfold.errors import (
    AllocatorError,
    AllocatorInvalidTargetError,
    AllocatorNoFeasibleError,
    CacheError,
    CacheMissingLayerError,
    CacheShapeError,
    CacheValidationError,
    DeviceError,
    DTypeError,
    KernelError,
    KernelNotAvailableError,
    KVCompressError,
    MethodConfigError,
    MethodError,
    ShapeError,
    UnsupportedMethodError,
    VLLMAPIDriftError,
    VLLMError,
    VLLMNotAvailableError,
)


def test_base_hierarchy() -> None:
    assert issubclass(KVCompressError, Exception)
    for cls in (
        CacheError,
        CacheValidationError,
        CacheMissingLayerError,
        CacheShapeError,
        AllocatorError,
        AllocatorNoFeasibleError,
        AllocatorInvalidTargetError,
        VLLMError,
        VLLMNotAvailableError,
        VLLMAPIDriftError,
        MethodError,
        UnsupportedMethodError,
        MethodConfigError,
        ShapeError,
        DTypeError,
        DeviceError,
        KernelError,
        KernelNotAvailableError,
    ):
        assert issubclass(cls, KVCompressError), cls


def test_kvcompress_error_code_and_hint() -> None:
    err = KVCompressError("boom", hint="try again")
    assert err.code == "kvfold_error"
    assert err.message == "boom"
    assert err.hint == "try again"
    assert "boom" in str(err)
    assert "try again" in str(err)


def test_cache_validation_error_carries_shapes() -> None:
    err = CacheValidationError(layer=3, expected=(8, 1024, 128), actual=(8, 512, 128))
    assert err.layer == 3
    assert err.expected == (8, 1024, 128)
    assert err.actual == (8, 512, 128)
    assert err.code == "cache_validation_error"


def test_cache_missing_layer_error() -> None:
    err = CacheMissingLayerError(layer=7)
    assert err.layer == 7
    assert "7" in err.message
    assert err.code == "cache_missing_layer_error"


def test_allocator_no_feasible_error() -> None:
    err = AllocatorNoFeasibleError(target_ratio=8.0, achieved_ratio=3.5)
    assert err.target_ratio == 8.0
    assert err.achieved_ratio == 3.5
    assert "8.000" in err.message
    assert "3.500" in err.message
    assert err.code == "allocator_no_feasible_error"


def test_allocator_invalid_target_error() -> None:
    err = AllocatorInvalidTargetError(target_ratio=0.5)
    assert err.target_ratio == 0.5
    assert err.minimum == 1.0
    assert err.code == "allocator_invalid_target_error"


def test_vllm_not_available_error() -> None:
    err = VLLMNotAvailableError()
    assert "kvfold[vllm]" in (err.hint or "")
    assert err.code == "vllm_not_available_error"


def test_vllm_api_drift_error() -> None:
    err = VLLMAPIDriftError(missing="transfer_async")
    assert err.missing == "transfer_async"
    assert "transfer_async" in err.message
    assert err.code == "vllm_api_drift_error"


def test_unsupported_method_error() -> None:
    err = UnsupportedMethodError(method="weird", supported=("jolt", "flash"))
    assert err.method == "weird"
    assert err.supported == ("jolt", "flash")
    assert "weird" in err.message
    assert "jolt" in err.message
    assert err.code == "unsupported_method_error"


def test_method_config_error() -> None:
    err = MethodConfigError(method="jolt", field="ratio", message="must be >= 1.0")
    assert err.method == "jolt"
    assert err.field == "ratio"
    assert err.detail == "must be >= 1.0"
    assert err.code == "method_config_error"


def test_kernel_not_available_error() -> None:
    err = KernelNotAvailableError(backend="triton", required="CUDA")
    assert err.backend == "triton"
    assert err.required == "CUDA"
    assert err.code == "kernel_not_available_error"


def test_to_dict_round_trip() -> None:
    err = CacheValidationError(layer=1, expected=(2, 3), actual=(4, 5))
    blob = err.to_dict()
    assert blob == {
        "code": "cache_validation_error",
        "message": err.message,
        "hint": err.hint,
    }


def test_error_codes_unique() -> None:
    """Every concrete error has a unique code."""
    codes: list[str] = []
    for cls in (
        CacheError,
        CacheValidationError,
        CacheMissingLayerError,
        CacheShapeError,
        AllocatorError,
        AllocatorNoFeasibleError,
        AllocatorInvalidTargetError,
        VLLMError,
        VLLMNotAvailableError,
        VLLMAPIDriftError,
        MethodError,
        UnsupportedMethodError,
        MethodConfigError,
        ShapeError,
        DTypeError,
        DeviceError,
        KernelError,
        KernelNotAvailableError,
    ):
        codes.append(cls.code)
    assert len(codes) == len(set(codes)), f"duplicate codes: {codes}"


@pytest.mark.parametrize(
    "exc",
    [
        KVCompressError("x"),
        CacheError("x"),
        CacheShapeError("x"),
        AllocatorError("x"),
        VLLMError("x"),
        MethodError("x"),
        ShapeError("x"),
        DTypeError("x"),
        DeviceError("x"),
        KernelError("x"),
    ],
)
def test_caught_by_base(exc: KVCompressError) -> None:
    try:
        raise exc
    except KVCompressError as caught:
        assert caught is exc
    else:
        pytest.fail("KVCompressError did not catch the subclass")


def test_typo_in_import_is_a_typo() -> None:
    """Guard against a future rename accidentally exposing an old alias."""
    with pytest.raises(ImportError):
        from kvfold.errors import NotAnError  # noqa: F401
