"""Tests for benchmark measurement contracts and provenance helpers."""

from __future__ import annotations

import pytest

from kvfold.bench.provenance import environment_metadata
from kvfold.bench.speed import _time_call_stats, run_speed_sweep


def test_benchmark_metadata_records_runtime_contract() -> None:
    metadata = environment_metadata(seed=7, warmup=1, iterations=3)
    assert metadata["seed"] == 7
    assert metadata["warmup_iterations"] == 1
    assert metadata["timed_iterations"] == 3
    assert metadata["python"]
    assert metadata["torch"]


def test_speed_measurement_requires_variance_sample() -> None:
    with pytest.raises(ValueError, match=r"n_iter.*2"):
        _time_call_stats(lambda: None, n_warmup=0, n_iter=1)


def test_speed_rows_include_variance_and_extrema() -> None:
    rows = run_speed_sweep(m=2, T=8, dh=4, ratio=2.0, iterations=2, warmup=0)
    assert rows
    for row in rows:
        assert row["compress_std_ms"] >= 0
        assert row["compress_ci95_ms"] >= 0
        assert row["compress_min_ms"] <= row["compress_max_ms"]
        assert row["decompress_min_ms"] <= row["decompress_max_ms"]
