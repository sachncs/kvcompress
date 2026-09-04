"""Paper reproduction scripts.

Modules:

- :mod:`kvfold.paper.table2` — Table 2 reconstruction-error benchmarks.
- :mod:`kvfold.paper.perplexity` — Table 1 WikiText perplexity sweep.
- :mod:`kvfold.paper.longbench` — long-context evaluation (research-only).

Each module is importable; the CLI exposes them as ``kvfold paper {table1, table2, longbench}``.
"""

from __future__ import annotations

__all__ = ["perplexity", "table2", "longbench"]
