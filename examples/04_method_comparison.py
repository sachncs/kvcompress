"""04 — Comparison: JoLT vs low-rank vs int4 on synthetic K/V.

Runs the same K/V through several methods and reports relative Frobenius
error, bytes occupied, and the compression ratio achieved.
"""

from __future__ import annotations

import torch

from kvfold import (
    Flash,
    IntQuant,
    Jolt,
    Low,
)


def rel_err(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return float(torch.linalg.norm(x - x_hat) / torch.linalg.norm(x))


def main() -> None:
    torch.manual_seed(0)
    K = torch.randn(8, 256, 64)
    V = torch.randn(8, 256, 64)
    original_bytes = K.numel() * K.element_size() * 2

    methods: list[tuple[str, object]] = [
        ("jolt @ 3x", Jolt(ratio=3.0, bits=(0, 4, 8))),
        ("flash @ 3x", Flash(ratio=3.0, bits=(0, 4, 8))),
        ("low @ rank=32", Low(rank=32)),
        ("int4 per-channel", IntQuant(bits=4, per_channel=True)),
    ]

    print(f"{'method':<24} {'K err':>10} {'V err':>10} {'bytes':>12} {'ratio':>10}")
    print("-" * 70)
    for name, comp in methods:
        kp, vp = comp.compress(K, V)
        k_hat, v_hat = comp.restore(kp, vp)
        e_k = rel_err(K, k_hat)
        e_v = rel_err(V, v_hat)
        bytes_used = kp.bytes_compressed + vp.bytes_compressed
        ratio = original_bytes / bytes_used
        print(f"{name:<24} {e_k:>10.4f} {e_v:>10.4f} {bytes_used:>12} {ratio:>9.2f}x")


if __name__ == "__main__":
    main()
