"""Compression / decompression speed orchestrator."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from kvfold.benchmarks.throughput import run_speed_sweep

log = logging.getLogger(__name__)


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
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
