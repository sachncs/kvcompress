"""Command-line interface for kvcompress.

Implemented in M9.
"""

from __future__ import annotations

import typer

app = typer.Typer(help="kvcompress CLI")


@app.command()
def version() -> None:
    """Print the kvcompress version."""
    from kvcompress import __version__

    typer.echo(__version__)


@app.command()
def validate() -> None:
    """Run a smoke test of the installed package."""
    typer.echo("kvcompress validate: not implemented yet (M9)")
    raise typer.Exit(code=1)


@app.command()
def benchmark() -> None:
    """Run benchmark suite."""
    typer.echo("kvcompress benchmark: not implemented yet (M9)")
    raise typer.Exit(code=1)


@app.command()
def profile() -> None:
    """Profile a model with compression enabled."""
    typer.echo("kvcompress profile: not implemented yet (M9)")
    raise typer.Exit(code=1)


@app.command()
def compress() -> None:
    """Run a one-shot compression pass on a prompt."""
    typer.echo("kvcompress compress: not implemented yet (M9)")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()