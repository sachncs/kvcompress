from __future__ import annotations

import pytest
import torch

from kvfold.core.float8 import Float8, e4m3_dtype


def test_float8_dtype_is_available_in_supported_torch() -> None:
    assert e4m3_dtype() == torch.float8_e4m3fn


def test_float8_round_trip_has_expected_shape() -> None:
    compressor = Float8(per_channel=True)
    key = torch.randn(2, 4, 8, dtype=torch.float32)
    payload, _ = compressor.compress(key, key)
    restored, _ = compressor.restore(payload, payload)
    assert restored.shape == key.shape
    assert restored.dtype == key.dtype
    assert torch.isfinite(restored).all()


def test_float8_group_size_validates_shape() -> None:
    compressor = Float8(per_channel=False, group_size=3)
    with pytest.raises(Exception, match="not divisible"):
        compressor.compress(torch.randn(2, 4, 8), torch.randn(2, 4, 8))
