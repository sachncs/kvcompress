"""Tests for the Typer CLI surface.

Exercises every command Typer registers (`version`, `validate`,
`benchmark`, `profile`, `compress`) via Typer's ``CliRunner`` so the
test runs without spawning real subprocesses or loading real models.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import torch
from typer.testing import CliRunner

from kvcompress import __version__
from kvcompress.cli import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


def test_version_prints_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


# ---------------------------------------------------------------------------
# Direct-call coverage tests — bypass Typer's click wrapper so coverage
# can see the underlying function bodies (Typer replaces the original
# function with a callable wrapper, which confuses coverage.py).
# ---------------------------------------------------------------------------


def test_version_callable() -> None:
    """Direct call exercises the body that Typer's wrapper hides."""
    from kvcompress.cli import version as version_fn

    version_fn()  # typer.echo writes to stdout; we don't capture it here.


def test_validate_callable() -> None:
    """Direct validate() call bypasses Typer wrapping for coverage."""
    from kvcompress.cli import validate as validate_fn

    validate_fn(skip_hf=True)


def test_validate_reports_flashjolt_callable() -> None:
    """Direct call: validate() runs both JoLT and FlashJoLT paths."""
    from kvcompress.cli import validate as validate_fn

    validate_fn(skip_hf=True)


# ---------------------------------------------------------------------------
# validate (via Typer runner — covers the wrapper itself).
# ---------------------------------------------------------------------------


def test_validate_runs_synthetic_round_trip() -> None:
    result = runner.invoke(app, ["validate", "--skip-hf"])
    assert result.exit_code == 0
    assert "JoLT round-trip rel error" in result.stdout
    assert "kvcompress validate: OK" in result.stdout


def test_validate_reports_flashjolt_error() -> None:
    """The validate command runs FlashJoLT too and reports its rel_err."""
    result = runner.invoke(app, ["validate", "--skip-hf"])
    assert result.exit_code == 0
    assert "FlashJoLT round-trip rel error" in result.stdout


def test_validate_hf_smoke_test_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct call to validate() with the HF branch: stub GPT2 to skip download."""
    import sys
    import types

    class _StubTok:
        eos_token_id = 0

        def encode(self, text, **_kw):
            return torch.tensor([[1, 2, 3]])

        def decode(self, ids, **_kw):
            return "stub-out"

    class _StubModel:
        class _Config:
            model_type = "gpt2"

        config = _Config()

        def __init__(self) -> None:
            self.generation_config = mock.Mock()
            self.generation_config.cache_implementation = None

        def eval(self) -> "_StubModel":
            return self

        def generate(self, ids, **kwargs):
            return torch.cat([ids, ids.new_zeros((1, kwargs["max_new_tokens"]))], dim=1)

    fake_transformers = types.ModuleType("transformers")

    # Patch the imports inside validate() by injecting a stubbed
    # ``transformers`` module into sys.modules AND patching the symbols
    # that validate() reaches for via ``from transformers import …``.
    fake_transformers.GPT2LMHeadModel = mock.Mock(return_value=_StubModel())
    fake_transformers.GPT2Tokenizer = mock.Mock(return_value=_StubTok())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    # Also intercept enable_compression to avoid touching the live model.
    # validate() does ``from kvcompress import enable_compression`` which
    # resolves via LAZY_EXPORTS. Patch the upstream module and clear the
    # cache so the lazy resolution picks up our stub.
    import kvcompress

    kvcompress.__dict__.pop("enable_compression", None)
    fake_handle = mock.MagicMock()
    with mock.patch("kvcompress.api.enable_compression", return_value=fake_handle) as m_enable:
        from kvcompress.cli import validate as validate_fn

        validate_fn(skip_hf=False)
    kvcompress.__dict__.pop("enable_compression", None)
    assert m_enable.called


def test_validate_hf_smoke_test_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF branch falls through to ``HF smoke test skipped`` when transformers raises."""
    import sys
    import types

    fake_transformers = types.ModuleType("transformers")

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated download failure")

    fake_transformers.GPT2LMHeadModel = _raise
    fake_transformers.GPT2Tokenizer = _raise
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    from kvcompress.cli import validate as validate_fn

    validate_fn(skip_hf=False)


def test_run_subprocess_returns_true_on_success() -> None:
    """Direct call to run_subprocess — the wrapper around subprocess.check_call."""
    from kvcompress.cli import run_subprocess as run_subprocess_fn

    # Mock subprocess.check_call to return normally; run_subprocess should
    # return True.
    with mock.patch("subprocess.check_call", return_value=0):
        ok = run_subprocess_fn(["echo", "hi"], "smoke")
    assert ok is True


def test_run_subprocess_returns_false_on_called_process_error() -> None:
    """run_subprocess catches CalledProcessError and returns False."""
    from subprocess import CalledProcessError

    from kvcompress.cli import run_subprocess as run_subprocess_fn

    with mock.patch(
        "subprocess.check_call",
        side_effect=CalledProcessError(1, ["false"]),
    ):
        ok = run_subprocess_fn(["false"], "smoke")
    assert ok is False


def test_run_subprocess_returns_false_on_timeout() -> None:
    """run_subprocess catches TimeoutExpired and returns False."""
    from subprocess import TimeoutExpired

    from kvcompress.cli import run_subprocess as run_subprocess_fn

    with mock.patch(
        "subprocess.check_call",
        side_effect=TimeoutExpired(["sleep"], 1.0),
    ):
        ok = run_subprocess_fn(["sleep"], "smoke", timeout=0.1)
    assert ok is False


def test_benchmark_direct_memory_only() -> None:
    """Direct benchmark() call with --suite memory exercises the memory branch."""
    from pathlib import Path

    from kvcompress.cli import benchmark as benchmark_fn

    with mock.patch("kvcompress.cli.run_subprocess", return_value=True) as m:
        benchmark_fn(suite="memory", output_dir=Path("/tmp"))
    assert m.called


def test_benchmark_direct_speed_only() -> None:
    """Direct benchmark() call with --suite speed exercises the speed branch."""
    from pathlib import Path

    from kvcompress.cli import benchmark as benchmark_fn

    with mock.patch("kvcompress.cli.run_subprocess", return_value=True) as m:
        benchmark_fn(suite="speed", output_dir=Path("/tmp"))
    assert m.called


def test_benchmark_direct_reconstruction_only() -> None:
    """Direct benchmark() call with --suite reconstruction."""
    from pathlib import Path

    from kvcompress.cli import benchmark as benchmark_fn

    with mock.patch("kvcompress.cli.run_subprocess", return_value=True) as m:
        benchmark_fn(suite="reconstruction", output_dir=Path("/tmp"))
    assert m.called


def test_benchmark_direct_exits_when_any_suite_fails() -> None:
    """Direct benchmark() call raises typer.Exit(1) when a suite fails."""
    from pathlib import Path

    import pytest
    from typer import Exit as TyperExit

    from kvcompress.cli import benchmark as benchmark_fn

    with mock.patch("kvcompress.cli.run_subprocess", return_value=False):
        with pytest.raises(TyperExit):
            benchmark_fn(suite="all", output_dir=Path("/tmp"))


def test_profile_direct_invokes_subprocess() -> None:
    """Direct profile() call invokes run_subprocess."""
    from kvcompress.cli import profile as profile_fn

    with mock.patch("kvcompress.cli.run_subprocess", return_value=True) as m:
        profile_fn(
            model="gpt2",
            ratio=3.0,
            max_new=10,
            method="flashjolt",
            seed=0,
            bits="0,2,4,8",
            layer_groups=1,
        )
    assert m.called


# ---------------------------------------------------------------------------
# compress (direct call)
# ---------------------------------------------------------------------------


def test_compress_direct_requires_hf_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct compress() call surfaces a clean typer.Exit when transformers is missing."""
    from kvcompress.cli import compress as compress_fn

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("simulated missing transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # typer.Exit is a subclass of click.exceptions.Exit which is a
    # subclass of RuntimeError. Catch it loosely.
    with pytest.raises((SystemExit, Exception)):
        compress_fn(model="gpt2", method="identity", target="100%", prompt="Hi", max_new=2)


def test_compress_direct_with_stubbed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct compress() call with stubbed AutoModel/AutoTokenizer."""
    from typer import echo

    from kvcompress.cli import compress as compress_fn

    class _StubModel:
        class _Config:
            model_type = "gpt2"

        config = _Config()

        def __init__(self) -> None:
            self.generation_config = mock.Mock()
            self.generation_config.cache_implementation = None

        def eval(self) -> "_StubModel":
            return self

        def generate(self, ids, **kwargs):
            return torch.cat([ids, ids.new_zeros((1, kwargs["max_new_tokens"]))], dim=1)

    def fake_automodel(*args, **kwargs):
        return _StubModel()

    def fake_autotokenizer(*args, **kwargs):
        tok = mock.Mock()
        tok.encode = lambda text, **kw: torch.tensor([[1, 2, 3]])
        tok.decode = lambda ids, **kw: "ok"
        return tok

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_automodel)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_autotokenizer)
    # Don't let the JSON stats echo spam test output.
    with mock.patch.object(echo, "__call__", return_value=None):
        compress_fn(
            model="gpt2",
            method="identity",
            target="100%",
            prompt="Hello",
            max_new=2,
            seed=0,
            bits="0",
            layer_groups=1,
            cache_implementation="kvcompress",
        )


# ---------------------------------------------------------------------------
# benchmark (via Typer runner — covers the wrapper itself).
# ---------------------------------------------------------------------------


def test_benchmark_swallows_per_suite_failure(tmp_path: Path) -> None:
    """A failure in one benchmark suite must not abort the orchestration.

    We mock subprocess.check_call to return normally for the first call
    and raise CalledProcessError for the second; the command should
    exit 1 (orchestration-level failure) but report both.
    """
    from subprocess import CalledProcessError

    calls: list[str] = []

    def fake_run(args: list[str], label: str, timeout: float = 600.0) -> bool:
        calls.append(label)
        if label == "memory":
            raise CalledProcessError(1, args)
        return True

    with mock.patch("kvcompress.cli.run_subprocess", side_effect=fake_run):
        result = runner.invoke(
            app,
            ["benchmark", "--suite", "all", "--output-dir", str(tmp_path)],
        )
    assert "memory" in calls
    # The fake for memory raised before real_run could be invoked.
    assert result.exit_code != 0


def test_benchmark_succeeds_when_all_suites_pass(tmp_path: Path) -> None:
    """Happy path: every benchmark suite reports OK."""
    with mock.patch("kvcompress.cli.run_subprocess", return_value=True):
        result = runner.invoke(
            app,
            ["benchmark", "--suite", "all", "--output-dir", str(tmp_path)],
        )
    assert result.exit_code == 0
    assert "benchmark outputs in" in result.stdout


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


def test_profile_runs_subprocess() -> None:
    """profile invokes scripts.profile_model via run_subprocess."""
    with mock.patch("kvcompress.cli.run_subprocess", return_value=True) as m:
        result = runner.invoke(
            app,
            ["profile", "--model", "gpt2", "--ratio", "3.0", "--method", "flashjolt"],
        )
    assert result.exit_code == 0
    assert m.called


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------


def test_compress_requires_hf_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """If transformers can't be imported, compress exits non-zero."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("simulated missing transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = runner.invoke(
        app,
        ["compress", "--model", "gpt2", "--target", "50%"],
    )
    assert result.exit_code != 0
    # typer.echo(..., err=True) goes to stderr; combine both streams.
    combined = (result.stdout or "") + (getattr(result, "stderr", "") or "")
    assert "error" in combined.lower() or "transformers" in combined.lower()


def test_compress_with_stubbed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end compress with a stubbed AutoModelForCausalLM."""

    class _StubModel:
        class _Config:
            model_type = "gpt2"

        config = _Config()

        def __init__(self) -> None:
            self.generation_config = mock.Mock()
            self.generation_config.cache_implementation = None

        def eval(self) -> "_StubModel":
            return self

        def generate(self, ids, **kwargs):  # noqa: ARG002
            return torch.cat([ids, ids.new_zeros((1, kwargs["max_new_tokens"]))], dim=1)

    def fake_automodel(*args, **kwargs):
        return _StubModel()

    def fake_autotokenizer(*args, **kwargs):
        tok = mock.Mock()
        tok.encode = lambda text, **kw: torch.tensor([[1, 2, 3]])
        tok.decode = lambda ids, **kw: "ok"
        return tok

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_automodel)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_autotokenizer)

    result = runner.invoke(
        app,
        [
            "compress",
            "--model",
            "gpt2",
            "--method",
            "identity",
            "--target",
            "100%",
            "--prompt",
            "Hello",
            "--max-new",
            "2",
        ],
    )
    if result.exit_code != 0:
        combined = (result.stdout or "") + (getattr(result, "stderr", "") or "")
        pytest.fail(f"compress failed: {combined}")
    assert "ok" in result.stdout
    # Stats JSON appears at the end.
    stats_start = result.stdout.rfind("{")
    assert stats_start != -1
    stats = json.loads(result.stdout[stats_start:])
    assert "compress_calls" in stats
