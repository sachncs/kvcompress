"""Table 2 reconstruction-error benchmark (paper-faithful synthetic K/V).

Runs JoLT, Flash, Low, and IntQuant on the paper's synthetic K/V at a
range of compression ratios and records per-method ``rel_err_K``,
``rel_err_V``, and ``bytes_K`` / ``bytes_V``.

This is a CPU-friendly sanity check; it does not load any HF model. The
synthetic spectrum mirrors Mistral-7B layer 15 (from the paper's Table
2 column), with the 1/i decay on both token and feature modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch

from kvfold.api import build_compressor

log = logging.getLogger(__name__)


@dataclass
class ReconstructionResult:
    method: str
    bits: int
    rel_err_K: float
    rel_err_V: float
    bytes_K: int
    bytes_V: int
    compression_ratio: float


def make_synthetic_kv(
    *,
    m: int = 8,
    length: int = 1024,
    head_dim: int = 128,
    k_token_rank: int = 228,
    v_token_rank: int = 563,
    k_feature_rank: int = 101,
    v_feature_rank: int = 126,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build synthetic K/V with a 1/i-decay Tucker core (Mistral-7B layer 15)."""
    torch.manual_seed(seed)
    k_core = torch.zeros(k_token_rank, k_feature_rank)
    for i in range(k_token_rank):
        for j in range(k_feature_rank):
            k_core[i, j] = 1.0 / (1 + (i + j))
    v_core = torch.zeros(v_token_rank, v_feature_rank)
    for i in range(v_token_rank):
        for j in range(v_feature_rank):
            v_core[i, j] = 1.0 / (1 + (i + j))
    k_token = torch.randn(length, k_token_rank)
    k_feature = torch.randn(head_dim, k_feature_rank)
    v_token = torch.randn(length, v_token_rank)
    v_feature = torch.randn(head_dim, v_feature_rank)
    K = torch.einsum("ta,dr,ar->td", k_token, k_feature, k_core).unsqueeze(0).expand(m, length, head_dim).contiguous()
    V = torch.einsum("ta,dr,ar->td", v_token, v_feature, v_core).unsqueeze(0).expand(m, length, head_dim).contiguous()
    return K, V


def rel_err(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Relative Frobenius error ``||x - x_hat||F / ||x||F``."""
    num = float(torch.linalg.norm(x - x_hat))
    den = float(torch.linalg.norm(x))
    if den == 0.0:
        return 0.0
    return num / den


def run_table2(
    *,
    m: int = 8,
    length: int = 1024,
    head_dim: int = 128,
    seed: int = 0,
    compression_ratio: float = 2.0,
    bits: int = 4,
) -> list[ReconstructionResult]:
    K, V = make_synthetic_kv(m=m, length=length, head_dim=head_dim, seed=seed)
    methods: list[tuple[str, dict[str, int | float]]] = [
        ("jolt", {"ratio": compression_ratio}),
        ("flash", {"ratio": compression_ratio}),
        ("low", {"rank": 64}),
        ("int4", {"bits": bits, "per_channel": True}),
    ]
    results: list[ReconstructionResult] = []
    original_bytes = K.numel() * K.element_size()
    for method, kwargs in methods:
        c = build_compressor(method, **kwargs)
        kp, vp = c.compress(K, V)
        kr, vr = c.restore(kp, vp)
        results.append(
            ReconstructionResult(
                method=method,
                bits=bits if "int" in method else 0,
                rel_err_K=rel_err(K, kr),
                rel_err_V=rel_err(V, vr),
                bytes_K=kp.bytes_compressed,
                bytes_V=vp.bytes_compressed,
                compression_ratio=original_bytes / max(kp.bytes_compressed, 1),
            )
        )
    return results


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Table 2 reconstruction benchmark")
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=2.0)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_table2(m=args.m, length=args.length, head_dim=args.head_dim, seed=args.seed, compression_ratio=args.ratio, bits=args.bits)
    rows = [r.__dict__ for r in results]
    print(f"{'method':<8} {'bits':>5} {'rel_err_K':>12} {'rel_err_V':>12} {'bytes_K':>10} {'ratio':>10}")
    for r in results:
        print(f"{r.method:<8} {r.bits:>5} {r.rel_err_K:>12.4f} {r.rel_err_V:>12.4f} {r.bytes_K:>10} {r.compression_ratio:>10.2f}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)


__all__ = ["ReconstructionResult", "make_synthetic_kv", "rel_err", "run_table2", "main"]
