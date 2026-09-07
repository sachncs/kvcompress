"""Run the paper Table 2 reproduction on synthetic K/V.

Usage:
    python -m scripts.run_table2_reconstruction --T 1024 --ratio 2.0 --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from kvfold.benchmarks.reconstruction import run_table2

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Table 2 reproduction")
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--dh", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=2.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rows = run_table2(m=args.m, T=args.T, dh=args.dh, seed=args.seed, compression_ratio=args.ratio)

    print(f"{'method':<24} {'K error':>10} {'V error':>10} {'ratio':>10}")
    print("-" * 60)
    out_rows = []
    for r in rows:
        print(
            f"{r.method:<24} {r.rel_err_K:>10.4f} {r.rel_err_V:>10.4f} {r.compression_ratio:>9.2f}x"
        )
        out_rows.append(
            {
                "method": r.method,
                "rel_err_K": r.rel_err_K,
                "rel_err_V": r.rel_err_V,
                "bytes_K": r.bytes_K,
                "bytes_V": r.bytes_V,
                "compression_ratio": r.compression_ratio,
            }
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out_rows, f, indent=2)
        log.info("wrote %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
