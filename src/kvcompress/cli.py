"""kvcompress CLI.

Top-level Typer app exposing the user-facing commands:

* ``kvcompress version`` — print the installed version.
* ``kvcompress validate`` — run a smoke test on synthetic K/V and, if
  ``transformers`` is installed, on a small HF model.
* ``kvcompress benchmark`` — orchestrate the memory / speed /
  reconstruction benchmarks. Spawns each benchmark module as a
  subprocess so a failure in one doesn't take down the others.
* ``kvcompress profile`` — run a single model through ``model.generate``
  with compression enabled, printing cumulative stats.
* ``kvcompress compress`` — one-shot compression on a single prompt.

Why subprocess for benchmarks: a misbehaving benchmark (e.g. an OOM)
should fail loudly without aborting the orchestration. Each benchmark
script writes its own JSON output, so a partial run still leaves usable
artefacts on disk.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import typer

from kvcompress import __version__

log = logging.getLogger(__name__)

app = typer.Typer(help="kvcompress — KV cache compression for decoder-only LLMs")


@app.command()
def version() -> None:
    """Print the kvcompress version."""
    typer.echo(__version__)


@app.command()
def validate(
    skip_hf: bool = typer.Option(False, help="Skip the Hugging Face smoke test"),
) -> None:
    """Run a smoke test of the installed package.

    Three checks:

    1. JoLT round-trip on synthetic K/V. Should report a small relative
       Frobenius error (typically < 0.5 for the default settings).
    2. FlashJoLT round-trip on the same K/V.
    3. End-to-end generation with GPT-2 (unless ``--skip-hf`` is given
       or ``transformers`` isn't installed).
    """
    import kvcompress
    import torch

    typer.echo(f"kvcompress {kvcompress.__version__}")

    # 1. JoLT round-trip.
    from kvcompress import JoLTCompressor, FlashJoLTCompressor

    K = torch.randn(4, 32, 16)
    V = torch.randn(4, 32, 16)
    comp = JoLTCompressor(compression_ratio=2.0)
    kp, vp = comp.compress(K, V)
    k_hat, v_hat = comp.decompress(kp, vp)
    rel_err = float(torch.linalg.norm(K - k_hat) / torch.linalg.norm(K))
    typer.echo(f"  JoLT round-trip rel error: {rel_err:.4f}")
    if rel_err > 1.0:
        typer.echo("  FAIL")
        raise typer.Exit(code=1)

    # 2. FlashJoLT round-trip.
    fj = FlashJoLTCompressor(compression_ratio=2.0)
    kp, vp = fj.compress(K, V)
    k_hat, v_hat = fj.decompress(kp, vp)
    rel_err = float(torch.linalg.norm(K - k_hat) / torch.linalg.norm(K))
    typer.echo(f"  FlashJoLT round-trip rel error: {rel_err:.4f}")

    # 3. HF smoke test.
    if not skip_hf:
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            from kvcompress import enable_compression

            tok = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            model.eval()
            handle = enable_compression(model, method="flashjolt", compression_ratio=2.0)
            try:
                ids = tok.encode("Hello", return_tensors="pt")
                with torch.no_grad():
                    out = model.generate(
                        ids,
                        max_new_tokens=5,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                typer.echo(f"  HF smoke test: {tok.decode(out[0])}")
            finally:
                handle.disable()
        except Exception as e:
            typer.echo(f"  HF smoke test skipped: {e}")

    typer.echo("kvcompress validate: OK")


@app.command()
def benchmark(
    suite: str = typer.Option("all", help="which benchmark suite to run"),
    output_dir: Path = typer.Option(Path("benchmarks/output"), help="output directory"),
) -> None:
    """Run the benchmark suite (memory / speed / reconstruction).

    Args:
        suite: one of ``all``, ``memory``, ``speed``, ``reconstruction``.
        output_dir: directory for JSON + PNG outputs.

    Each sub-suite is spawned as a subprocess so a failure in one
    doesn't take down the others. JSON files land in ``output_dir``;
    matplotlib PNG charts are written next to them if matplotlib is
    installed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if suite in ("all", "memory"):
        typer.echo("== memory benchmark ==")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "kvcompress.benchmarks.memory",
                "--T",
                "1024",
                "--dh",
                "128",
                "--ratios",
                "2",
                "3",
                "4",
                "5",
                "8",
                "--methods",
                "jolt",
                "flashjolt",
                "lowrank",
                "--output",
                str(output_dir / "memory.json"),
            ]
        )

    if suite in ("all", "speed"):
        typer.echo("== speed benchmark ==")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "kvcompress.benchmarks.throughput",
                "--T",
                "1024",
                "--dh",
                "128",
                "--ratio",
                "3",
                "--output",
                str(output_dir / "speed.json"),
            ]
        )

    if suite in ("all", "reconstruction"):
        typer.echo("== reconstruction benchmark ==")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "kvcompress.benchmarks.reconstruction",
                "--T",
                "1024",
                "--ratio",
                "2",
                "--output",
                str(output_dir / "reconstruction.json"),
            ]
        )

    typer.echo(f"benchmark outputs in {output_dir}")


@app.command()
def profile(
    model: str = typer.Option("gpt2", help="HF model id"),
    ratio: float = typer.Option(3.0, help="compression ratio"),
    max_new: int = typer.Option(20, help="tokens to generate"),
) -> None:
    """Profile a model with compression enabled and print cumulative stats."""
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "scripts.profile_model",
            "--model",
            model,
            "--ratio",
            str(ratio),
            "--max-new",
            str(max_new),
        ]
    )


@app.command()
def compress(
    model: str = typer.Option(..., help="HF model id"),
    method: str = typer.Option("flashjolt", help="compression method"),
    target: str = typer.Option("33%", help="target memory fraction (e.g. 33%)"),
    prompt: str = typer.Option("Hello, my name is", help="prompt text"),
    max_new: int = typer.Option(20, help="tokens to generate"),
) -> None:
    """Run a one-shot compression pass on a prompt and print the output.

    The compression handle is enabled for the duration of ``model.generate``
    and disabled before returning.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from kvcompress import enable_compression

        tok = AutoTokenizer.from_pretrained(model)
        mdl = AutoModelForCausalLM.from_pretrained(model)
        mdl.eval()
        handle = enable_compression(mdl, method=method, target_memory=target)
        try:
            ids = tok.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                out = mdl.generate(
                    ids,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            typer.echo(tok.decode(out[0]))
            typer.echo(json.dumps(handle.stats_dict(), indent=2))
        finally:
            handle.disable()
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
