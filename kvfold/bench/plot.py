"""Plot helpers for benchmark scripts.

Uses matplotlib (optional, in the ``bench`` extra) and saves PNG figures
next to the benchmark output JSON. When matplotlib is not installed
each function returns silently rather than raising — benchmarks that
write JSON still complete; only the plot step is skipped.

We don't pin matplotlib to a version: the API surface used here
(``plt.subplots``, ``ax.bar``, ``ax.set_yscale``, ``fig.savefig``) has
been stable since matplotlib 3.x.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def have_matplotlib() -> bool:
    """Return True if ``matplotlib`` is importable."""
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


def plot_memory_sweep(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Bar chart: bytes_compressed vs ratio_target, grouped by method.

    Args:
        rows: list of dicts as written by :func:`run_memory_sweep`.
        output_path: destination PNG path.

    Skips silently if matplotlib isn't installed.
    """
    if not have_matplotlib():
        log.warning("matplotlib not installed; skipping plot")
        return
    import matplotlib.pyplot as plt

    methods = sorted({r["method"] for r in rows})
    ratios = sorted({r["ratio_target"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        ys = [
            next(
                (
                    r["bytes_compressed"]
                    for r in rows
                    if r["method"] == m and r["ratio_target"] == r_target
                ),
                0,
            )
            for r_target in ratios
        ]
        xs = [j + i * width for j in range(len(ratios))]
        ax.bar(xs, ys, width=width, label=m)
    ax.set_xticks([j + width * (len(methods) - 1) / 2 for j in range(len(ratios))])
    ax.set_xticklabels([str(r) for r in ratios])
    ax.set_xlabel("Target compression ratio")
    ax.set_ylabel("Bytes (compressed)")
    ax.set_title("Memory benchmark")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_reconstruction_table(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Bar chart: relative Frobenius error (K and V) per method.

    Y-axis is log scale because the spread across methods is large.
    Skips silently if matplotlib isn't installed.
    """
    if not have_matplotlib():
        log.warning("matplotlib not installed; skipping plot")
        return
    import matplotlib.pyplot as plt

    methods = [r["method"] for r in rows]
    k_err = [r["rel_err_K"] for r in rows]
    v_err = [r["rel_err_V"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    xs = list(range(len(methods)))
    ax.bar([x - width / 2 for x in xs], k_err, width=width, label="K error")
    ax.bar([x + width / 2 for x in xs], v_err, width=width, label="V error")
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("Relative Frobenius error")
    ax.set_yscale("log")
    ax.set_title("Reconstruction fidelity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def load_json(path: str | Path) -> Any:
    """Read a JSON file and return its parsed contents."""
    with open(path) as f:
        return json.load(f)


__all__ = [
    "load_json",
    "plot_memory_sweep",
    "plot_reconstruction_table",
]
