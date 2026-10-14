"""Compression / decompression speed benchmark.

Measures mean wall time for :meth:`Compressor.compress` and
:meth:`Compressor.decompress` across JoLT, Flash, and the low-rank
baseline. Includes warm-up iterations to amortise first-call cache effects.

Usage::

    python -m kvfold.bench.speed --T 1024 --dh 128 --ratio 3

On CPU the three methods are roughly comparable at small T; Flash
pulls ahead at T ≥ 1024 once the SVD becomes the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from math import sqrt
from statistics import mean, stdev

import torch

from kvfold.core.flash import Flash
from kvfold.core.jolt import Jolt
from kvfold.core.low import Low

log = logging.getLogger(__name__)


def time_call(fn, n_warmup: int = 1, n_iter: int = 5) -> float:
    """Return mean wall time in milliseconds across ``n_iter`` calls.

    Includes ``n_warmup`` unmeasured calls before the timed loop to
    amortise first-call cache effects (PyTorch's allocator, JIT
    compilation, etc.).
    """
    return _time_call_stats(fn, n_warmup=n_warmup, n_iter=n_iter)["mean_ms"]


def _time_call_stats(fn, *, n_warmup: int, n_iter: int) -> dict[str, float]:
    """Measure wall-time samples and return mean, deviation, and extrema."""
    if n_warmup < 0 or n_iter < 2:
        raise ValueError("n_warmup must be >= 0 and n_iter must be >= 2")
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": mean(times),
        "std_ms": stdev(times),
        "ci95_ms": 1.96 * stdev(times) / sqrt(len(times)),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def run_speed_sweep(
    *,
    m: int,
    T: int,
    dh: int,
    ratio: float,
    seed: int = 0,
    warmup: int = 1,
    iterations: int = 5,
) -> list[dict[str, object]]:
    """Measure compress / decompress wall time for each method at ``ratio``.

    Args:
        m: merged head × layer count.
        T: token axis length.
        dh: per-head feature dim.
        ratio: compression ratio passed to JoLT and Flash.
        seed: random seed for the synthetic K/V.
        warmup: unmeasured calls before timing.
        iterations: measured calls; must be at least two for deviation.

    Returns:
        List of result dicts; one per method.
    """
    torch.manual_seed(seed)
    K = torch.randn(m, T, dh)
    V = torch.randn(m, T, dh)
    rows = []

    comps = [
        ("jolt", Jolt(ratio=ratio)),
        ("flash", Flash(ratio=ratio)),
        ("low", Low(rank=max(1, int(min(m * T, dh) / ratio)))),
    ]
    for name, comp in comps:
        # Warm up the allocator.
        kp, vp = comp.compress(K, V)

        compress_stats = _time_call_stats(
            lambda compressor=comp: compressor.compress(K, V),
            n_warmup=warmup,
            n_iter=iterations,
        )
        kp, vp = comp.compress(K, V)
        decompress_stats = _time_call_stats(
            lambda compressor=comp, key_payload=kp, value_payload=vp: compressor.restore(
                key_payload, value_payload
            ),
            n_warmup=warmup,
            n_iter=iterations,
        )

        original_bytes = K.numel() * K.element_size() * 2
        compressed_bytes = kp.bytes_compressed + vp.bytes_compressed
        rows.append(
            {
                "method": name,
                "compress_ms": compress_stats["mean_ms"],
                "compress_std_ms": compress_stats["std_ms"],
                "compress_ci95_ms": compress_stats["ci95_ms"],
                "compress_min_ms": compress_stats["min_ms"],
                "compress_max_ms": compress_stats["max_ms"],
                "decompress_ms": decompress_stats["mean_ms"],
                "decompress_std_ms": decompress_stats["std_ms"],
                "decompress_ci95_ms": decompress_stats["ci95_ms"],
                "decompress_min_ms": decompress_stats["min_ms"],
                "decompress_max_ms": decompress_stats["max_ms"],
                "bytes_original": original_bytes,
                "bytes_compressed": compressed_bytes,
                "ratio_achieved": original_bytes / compressed_bytes,
            }
        )
    return rows


def main() -> None:  # pragma: no cover
    """CLI entry point. Parses ``--m --T --dh --ratio --seed --output``."""
    parser = argparse.ArgumentParser(description="Compression speed benchmark")
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--dh", type=int, default=128)
    parser.add_argument("--ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rows = run_speed_sweep(m=args.m, T=args.T, dh=args.dh, ratio=args.ratio, seed=args.seed)

    print(f"{'method':<14} {'compress_ms':>14} {'decompress_ms':>14} {'ratio':>10}")
    print("-" * 60)
    for r in rows:
        print(
            f"{r['method']:<14} {r['compress_ms']:>14.2f} "
            f"{r['decompress_ms']:>14.2f} {r['ratio_achieved']:>9.2f}x"
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
