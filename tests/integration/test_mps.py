"""Apple MPS integration coverage for the device-portable compressor path."""

from __future__ import annotations

import pytest
import torch

from kvfold.api import build_compressor

pytestmark = [pytest.mark.integration, pytest.mark.mps]


def _mps_device() -> torch.device:
    if not torch.backends.mps.is_available():
        pytest.skip("Apple MPS is not available")
    return torch.device("mps")


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("jolt", {"ratio": 2.0}),
        ("flash", {"ratio": 2.0}),
        ("low", {"rank": 8}),
        ("int2", {}),
        ("int4", {}),
        ("int8", {}),
        ("fp16", {}),
        ("bf16", {}),
        ("pass", {}),
    ],
)
def test_supported_compressor_round_trip_on_mps(method: str, kwargs: dict[str, object]) -> None:
    """Supported compressors preserve shape and device on MPS."""
    device = _mps_device()
    torch.manual_seed(0)
    key = torch.randn(4, 32, 16, device=device)
    value = torch.randn(4, 32, 16, device=device)

    compressor = build_compressor(method, **kwargs)
    key_payload, value_payload = compressor.compress(key, value)
    restored_key, restored_value = compressor.restore(key_payload, value_payload)

    assert restored_key.shape == key.shape
    assert restored_value.shape == value.shape
    assert restored_key.device.type == "mps"
    assert restored_value.device.type == "mps"


def test_fp8_is_reported_as_unsupported_on_mps() -> None:
    """MPS lacks Float8 storage support and must fail explicitly."""
    device = _mps_device()
    key = torch.randn(2, 8, 8, device=device)
    value = torch.randn(2, 8, 8, device=device)
    compressor = build_compressor("fp8")

    with pytest.raises((RuntimeError, TypeError), match=r"(?i)(mps|float8)"):
        compressor.compress(key, value)
