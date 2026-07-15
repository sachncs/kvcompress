"""Tests for the Typer CLI surface.

The CLI is thin (it spawns benchmark scripts as subprocesses), so the
tests focus on the high-level command dispatch, not the underlying
benchmark implementations (those have their own tests).
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from kvcompress import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_version_command(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    # The version line is the first non-empty line.
    output = result.stdout.strip()
    assert output.startswith("0."), f"unexpected version: {output!r}"


def test_version_uses_module_version(runner: CliRunner) -> None:
    import kvcompress

    result = runner.invoke(cli.app, ["version"])
    assert kvcompress.__version__ in result.stdout


def test_validate_help_lists_options(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["validate", "--help"])
    assert result.exit_code == 0
    assert "--skip-hf" in result.stdout


def test_benchmark_help(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--suite" in result.stdout


def test_profile_help(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["profile", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.stdout


def test_compress_help(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["compress", "--help"])
    assert result.exit_code == 0
    assert "--method" in result.stdout
    assert "--target" in result.stdout


def test_compress_help_lists_supported_methods(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["compress", "--help"])
    assert "flashjolt" in result.stdout
    assert "jolt" in result.stdout


def test_validate_skip_hf_runs_synthetic_check(runner: CliRunner) -> None:
    """``validate --skip-hf`` should run only the synthetic round-trip
    checks and exit 0."""
    result = runner.invoke(cli.app, ["validate", "--skip-hf"])
    assert result.exit_code == 0, result.stdout
    assert "JoLT round-trip" in result.stdout
    assert "FlashJoLT round-trip" in result.stdout
    assert "OK" in result.stdout
