#!/usr/bin/env python3
"""Thin wrapper around :mod:`kvfold.paper.perplexity`.

Usage::

    python scripts/run_table1_perplexity.py --model gpt2 --ratios 1 2 3 4 5 8 \
        --output results/perplexity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Table 1 perplexity sweep")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0, 8.0])
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--method", type=str, default="jolt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from kvfold.paper.perplexity import run_table1_perplexity

    rows = run_table1_perplexity(
        model_name=args.model,
        ratios=tuple(args.ratios),
        sequence_length=args.length,
        method=args.method,
        seed=args.seed,
    )
    for row in rows:
        print(f"ratio={row['ratio']:>5}  method={row['method']:<10}  ppl={row['perplexity']:.3f}  wall={row['wall_ms']:.1f}ms")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
