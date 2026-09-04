"""Tests for the YAML fixtures."""

from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")


def test_jolt_default_loads() -> None:
    from pathlib import Path

    data = yaml.safe_load(Path("tests/fixtures/jolt_default.yaml").read_text())
    assert data["method"] == "jolt"
    assert data["ratio"] == 3.0
    assert data["bits"] == [0, 2, 4, 8]
    assert data["distribution"] == "gaussian"


def test_flash_default_loads() -> None:
    from pathlib import Path

    data = yaml.safe_load(Path("tests/fixtures/flash_default.yaml").read_text())
    assert data["method"] == "flash"
    assert data["cap_policy"] == "linear"


def test_int4_per_channel_loads() -> None:
    from pathlib import Path

    data = yaml.safe_load(Path("tests/fixtures/int4_per_channel.yaml").read_text())
    assert data["method"] == "int4"
    assert data["bits"] == 4
    assert data["per_channel"] is True
