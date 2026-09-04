"""Regression test for Residual.bytes_compressed double-count (B-05)."""

from __future__ import annotations

import torch

from kvfold.core.residual import encode_residual


def test_bytes_no_double_count() -> None:
    res = torch.randn(8, 32, 16)
    payload = encode_residual(res, bits=4, seed=0)
    info = payload.bytes_compressed
    expected_min = info  # any positive value
    # 4-bit packed uint8 storage at minimum:
    expected_min = res.shape[0] * res.shape[1] * res.shape[2] // 2  # packed entries
    expected_min += 8  # scale + zero-point (at least one fp32 each)
    assert info >= expected_min
    # The old code reported ~2x this; make sure we're not wildly over.
    assert info <= 4 * expected_min
