from __future__ import annotations

import pytest
import torch

from kvfold.core.rank import Adaptive, Fixed, TailMass


def test_fixed_rank_uses_explicit_feature_rank() -> None:
    ranks = Fixed(r_token=5, r_feature=3).rank(torch.randn(2, 8, 4))
    assert ranks == (5, 3)


def test_adaptive_rank_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="target_ratio"):
        Adaptive(target_ratio=1.0)


def test_tail_mass_returns_bounded_ranks() -> None:
    ranks = TailMass().rank(torch.randn(2, 8, 4), epsilon=0.2)
    assert 1 <= ranks[0] <= 8
    assert 1 <= ranks[1] <= 4
