"""Benchmark scripts and orchestration.

Each module here is a Typer-style entry point that can also be invoked
via ``python -m kvcompress.benchmarks.<name>``. They are wired into the
top-level :mod:`kvcompress.cli` ``benchmark`` command.

* :mod:`.memory` — bytes occupied by the compressed cache at different
  ratios.
* :mod:`.throughput` — compress / decompress wall time per call.
* :mod:`.reconstruction` — relative Frobenius error at the requested
  ratio (mirrors paper Table 2).
* :mod:`.plot` — matplotlib bar charts for memory and reconstruction
  sweeps (no-op if matplotlib is not installed).
"""

__all__: list[str] = []
