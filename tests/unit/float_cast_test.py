from __future__ import annotations

import pytest
import torch

from kvfold.core.float_cast import FloatCast


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_float_cast_round_trip_preserves_shape_and_dtype(dtype: torch.dtype) -> None:
    key = torch.randn(2, 4, 8, dtype=torch.float32)
    value = torch.randn_like(key)
    compressor = FloatCast(dtype=dtype)
    key_payload, value_payload = compressor.compress(key, value)
    restored_key, restored_value = compressor.restore(key_payload, value_payload)
    assert key_payload.data["value"].dtype == dtype
    assert restored_key.dtype == key.dtype
    assert restored_value.dtype == value.dtype
    assert restored_key.shape == key.shape


def test_float_cast_rejects_unknown_options() -> None:
    with pytest.raises(Exception, match="unknown"):
        FloatCast(dtype=torch.float16, unknown=True)
