"""Compression / decompression speed benchmark.

Measures mean wall time for :meth:`KVCompressor.compress` and
:meth:`KVCompressor.decompress` across JoLT, FlashJoLT, and the low-rank
baseline. Includes warm-up iterations to amortise first-call cache effects.

Usage::

    python -m kvcompress.benchmarks.throughput --T 1024 --dh 128 --ratio 3

On CPU the three methods are roughly comparable at small T; FlashJoLT
pulls ahead at T ≥ 1024 once the SVD becomes the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch

from kvcompress.compressor.flashjolt import FlashJoLTCompressor
from kvcompress.compressor.jolt import JoLTCompressor
from kvcompress.compressor.lowrank import LowRankCompressor

log = logging.getLogger(__name__)


def time_call(fn, n_warmup: int = 1, n_iter: int = 5) -> float:
    """Return mean wall time in milliseconds across ``n_iter`` calls."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def run_speed_sweep(
    *,
    m: int,
    T: int,
    dh: int,
    ratio: float,
    seed: int = 0,
) -> list[dict[str, object]]:
    torch.manual_seed(seed)
    K = torch.randn(m, T, dh)
    V = torch.randn(m, T, dh)
    rows = []

    comps = [
        ("jolt", JoLTCompressor(compression_ratio=ratio)),
        ("flashjolt", FlashJoLTCompressor(compression_ratio=ratio)),
        ("lowrank", LowRankCompressor(rank=max(1, int(min(m * T, dh) / ratio)))),
    ]
    for name, comp in comps:
        # Warm up the allocator.
        kp, vp = comp.compress(K, V)

        compress_ms = time_call(lambda: comp.compress(K, V))
        kp, vp = comp.compress(K, V)
        decompress_ms = time_call(lambda: comp.decompress(kp, vp))

        original_bytes = K.numel() * K.element_size() * 2
        compressed_bytes = kp.bytes_compressed + vp.bytes_compressed
        rows.append(
            {
                "method": name,
                "compress_ms": compress_ms,
                "decompress_ms": decompress_ms,
                "bytes_original": original_bytes,
                "bytes_compressed": compressed_bytes,
                "ratio_achieved": original_bytes / compressed_bytes,
            }
        )
    return rows


def main() -> None:
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
            f"{r['method']:<14} {r['compress_ms']:>14.2f} {r['decompress_ms']:>14.2f} {r['ratio_achieved']:>9.2f}x"
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
