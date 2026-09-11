"""Memory benchmark orchestrator."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from kvfold.bench.memory import run_memory_sweep

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory benchmark")
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--dh", type=int, default=128)
    parser.add_argument("--ratios", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0, 8.0])
    parser.add_argument("--methods", type=str, nargs="+", default=["jolt", "flash", "low"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rows = run_memory_sweep(
        m=args.m,
        T=args.T,
        dh=args.dh,
        ratios=args.ratios,
        methods=args.methods,
        seed=args.seed,
    )

    print(f"{'method':<14} {'target':>10} {'orig(B)':>12} {'compressed(B)':>14} {'actual':>10}")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['method']:<14} {r['ratio_target']:>10.2f} {r['bytes_original']:>12} {r['bytes_compressed']:>14} {r['ratio_achieved']:>9.2f}x"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
