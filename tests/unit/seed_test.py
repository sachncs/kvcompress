from __future__ import annotations

import torch

from kvfold.runtime.seed import Seed


def test_seed_restores_cpu_rng_state() -> None:
    torch.manual_seed(123)
    before = torch.get_rng_state()
    with Seed(999):
        _ = torch.randn(4)
    after = torch.get_rng_state()
    assert torch.equal(before, after)


def test_seed_is_reproducible_inside_context() -> None:
    with Seed(7):
        first = torch.randn(4)
    with Seed(7):
        second = torch.randn(4)
    assert torch.equal(first, second)
