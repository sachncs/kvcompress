"""Regression test for SVD.randomise global RNG mutation (B-07)."""

from __future__ import annotations

import torch

from kvfold.core.svd import Randomized


def test_randomise_no_global_mutation() -> None:
    saved = torch.get_rng_state()
    try:
        d = Randomized(seed=42)
        d.decompose(torch.randn(64, 64), rank=8)
        d.decompose(torch.randn(64, 64), rank=8)
        current = torch.get_rng_state()
        assert torch.equal(saved, current), "SVD.randomise mutated the global RNG state"
    finally:
        torch.set_rng_state(saved)
