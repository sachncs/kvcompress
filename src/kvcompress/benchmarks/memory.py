"""Memory benchmark — bytes occupied by compressed vs uncompressed cache.

Compares :class:`~kvcompress.IdentityCompressor` against
:class:`~kvcompress.JoLTCompressor`,
:class:`~kvcompress.FlashJoLTCompressor`, and
:class:`~kvcompress.LowRankCompressor` across a sweep of compression
ratios. Reports the achieved bytes per method and ratio.

Usage::

    python -m kvcompress.benchmarks.memory --T 1024 --dh 128 --m 8 --ratio 3.0

The benchmark runs purely on synthetic tensors so it does not require
GPU resources.
"""

from __future__ import annotations

import argparse
import json
import logging

import torch

from kvcompress.compressor.flashjolt import FlashJoLTCompressor
from kvcompress.compressor.identity import IdentityCompressor
from kvcompress.compressor.jolt import JoLTCompressor
from kvcompress.compressor.lowrank import LowRankCompressor

log = logging.getLogger(__name__)


def run_memory_sweep(
    *,
    m: int,
    T: int,
    dh: int,
    ratios: list[float],
    methods: list[str],
    seed: int = 0,
) -> list[dict[str, object]]:
    torch.manual_seed(seed)
    K = torch.randn(m, T, dh)
    V = torch.randn(m, T, dh)
    rows: list[dict[str, object]] = []

    # Identity baseline.
    idc = IdentityCompressor()
    kp, vp = idc.compress(K, V)
    rows.append(
        {
            "method": "identity",
            "ratio_target": 1.0,
            "bytes_original": K.numel() * K.element_size() * 2,
            "bytes_compressed": kp.bytes_compressed + vp.bytes_compressed,
            "ratio_achieved": (K.numel() * K.element_size() * 2)
            / (kp.bytes_compressed + vp.bytes_compressed),
        }
    )

    for method in methods:
        for ratio in ratios:
            comp = None
            if method == "jolt":
                comp = JoLTCompressor(compression_ratio=ratio)
            elif method == "flashjolt":
                comp = FlashJoLTCompressor(compression_ratio=ratio)
            elif method == "lowrank":
                comp = LowRankCompressor(rank=max(1, int(min(m * T, dh) / ratio)))
            else:
                log.warning("unknown method %s; skipping", method)
                continue
            kp, vp = comp.compress(K, V)
            original_bytes = K.numel() * K.element_size() * 2
            compressed = kp.bytes_compressed + vp.bytes_compressed
            rows.append(
                {
                    "method": method,
                    "ratio_target": ratio,
                    "bytes_original": original_bytes,
                    "bytes_compressed": compressed,
                    "ratio_achieved": original_bytes / compressed,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory benchmark")
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--dh", type=int, default=128)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[2.0, 3.0, 4.0, 5.0, 8.0],
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["jolt", "flashjolt", "lowrank"],
    )
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
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
