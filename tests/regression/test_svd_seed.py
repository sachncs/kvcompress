"""Regression test for SVD.randomise global RNG mutation (B-07)."""

from __future__ import annotations

import torch

from kvfold.core.svd import Randomized


def test_randomise_no_global_mutation() -> None:
    saved = torch.get_rng_state()
    try:
        d = Randomized(seed=42)
        # The input tensors are built with a local Generator so the test
        # itself does not advance the global RNG; only Randomized.decompose
        # is left free to advance it. If ``decompose`` left global state
        # untouched (the documented invariant), saved == current.
        gen = torch.Generator()
        gen.manual_seed(0)
        x1 = torch.randn(64, 64, generator=gen)
        x2 = torch.randn(64, 64, generator=gen)
        d.decompose(x1, rank=8)
        d.decompose(x2, rank=8)
        current = torch.get_rng_state()
        assert torch.equal(saved, current), "SVD.randomise mutated the global RNG state"
    finally:
        torch.set_rng_state(saved)
