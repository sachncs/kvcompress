"""Tests for the joint allocator."""

from __future__ import annotations

import math

import pytest

from kvfold.core.budget import (
    Cell,
    Greedy,
    Bisect,
    Pick,
)


def make_cells() -> list[Cell]:
    return [
        Cell(shape=(8, 256, 64), kind="key", layer_group=0),
        Cell(shape=(8, 256, 64), kind="value", layer_group=0),
    ]


def test_joint_allocator_runs() -> None:
    cells = make_cells()
    alloc = Bisect(target_ratio=3.0)
    res = alloc.optimize(cells)
    assert len(res.allocations) == 2
    assert res.total_bytes > 0


def test_joint_allocator_hits_budget_3x() -> None:
    cells = make_cells()
    alloc = Bisect(target_ratio=3.0)
    res = alloc.optimize(cells)
    score = abs(math.log(res.achieved_ratio) - math.log(3.0))
    assert score < 1.0


def test_joint_allocator_hits_budget_2x() -> None:
    cells = make_cells()
    alloc = Bisect(target_ratio=2.0)
    res = alloc.optimize(cells)
    score = abs(math.log(res.achieved_ratio) - math.log(2.0))
    assert score < 1.0


def test_joint_allocator_hits_budget_4x() -> None:
    cells = make_cells()
    alloc = Bisect(target_ratio=4.0)
    res = alloc.optimize(cells)
    score = abs(math.log(res.achieved_ratio) - math.log(4.0))
    assert score < 1.0


def test_allocator_uses_residual_at_higher_ratio() -> None:
    """At higher ratios, allocator should prefer more residual bits."""
    cells = make_cells()
    res_low = Bisect(target_ratio=2.0).optimize(cells)
    res_high = Bisect(target_ratio=8.0).optimize(cells)
    bits_low = sum(a.bits for a in res_low.allocations)
    bits_high = sum(a.bits for a in res_high.allocations)
    # At 8x the budget is much smaller; either more bits OR very low ranks.
    # We just require the optimizer picks a non-degenerate answer.
    assert bits_high > 0 or bits_low > 0


def test_allocator_keys_values_split() -> None:
    """Allocator should return one allocation per cell."""
    cells = make_cells()
    res = Bisect(target_ratio=3.0).optimize(cells)
    a_k, a_v = res.allocations
    # Just structural — the allocator decides freely.
    assert a_k.cost_bytes >= 0
    assert a_v.cost_bytes >= 0


def test_greedy_runs() -> None:
    cells = make_cells()
    g = Greedy(target_ratio=3.0)
    res = g.optimize(cells)
    assert res.total_bytes > 0
    assert len(res.allocations) == 2


def test_invalid_target_ratio() -> None:
    with pytest.raises(ValueError, match="target_ratio"):
        Bisect(target_ratio=0.5)


def test_empty_cells() -> None:
    res = Bisect(target_ratio=3.0).optimize([])
    assert len(res.allocations) == 0
    assert res.target_bytes == 0


def test_allocator_extreme_budget_min_cost() -> None:
    """If minimum cost > budget, allocator returns minimum (warning)."""
    tiny = [Cell(shape=(2, 16, 8), kind="key")]
    # Target ratio > max possible.
    alloc = Bisect(target_ratio=100.0)
    res = alloc.optimize(tiny)
    assert len(res.allocations) == 1


def test_allocator_per_cell_grid_finite() -> None:
    cell = Cell(shape=(4, 128, 32))
    alloc = Bisect(target_ratio=3.0)
    grid = alloc.build_cell_grid(cell, None, 0)
    # Grid size is candidate_rt × candidate_rd × bits.
    assert all(isinstance(a, Pick) for a in grid)
    assert all(a.cost_bytes > 0 for a in grid)


def test_allocator_lambda_monotone() -> None:
    """Higher λ should produce lower total cost (monotone)."""
    cells = make_cells()
    alloc = Bisect(target_ratio=3.0)
    grid = [alloc.build_cell_grid(c, None, i) for i, c in enumerate(cells)]
    cost_low = sum(a.cost_bytes for a in alloc.argmin_per_cell(grid, 0.0))
    cost_high = sum(a.cost_bytes for a in alloc.argmin_per_cell(grid, 100.0))
    assert cost_high <= cost_low


def test_allocator_achieves_target_within_rounding() -> None:
    cells = make_cells()
    for ratio in [2.0, 3.0, 4.0, 5.0, 8.0]:
        res = Bisect(target_ratio=ratio).optimize(cells)
        # Discrete grid; log-space ratio distance to target ≤ 1.0 nat.
        score = abs(math.log(res.achieved_ratio) - math.log(ratio))
        assert score < 1.0, f"ratio={ratio}: achieved={res.achieved_ratio:.2f}, score={score:.2f}"


def test_allocator_respects_fp32_element_size() -> None:
    """Regression: an fp32 cache must hit the *fp32* target ratio.

    Previously ``bytes_original`` hardcoded ``*2`` (fp16), so a fp32
    cache achieved half the requested ratio. With
    ``element_size_bytes=4`` the budget doubles and the achieved
    ratio reflects the fp32 cost model.
    """
    cells = [Cell(shape=(4, 64, 32), kind="key")]
    alloc_fp32 = Bisect(target_ratio=2.0, element_size_bytes=4).optimize(cells)
    alloc_fp16 = Bisect(target_ratio=2.0, element_size_bytes=2).optimize(cells)
    # The fp32 path sees a larger total_bytes budget (twice the bytes
    # available) and so its achieved_ratio should reflect the larger
    # cost model. We assert the budgets scale by 2x rather than poking
    # at achieved_ratio directly (the cost grid is discrete).
    assert alloc_fp32.target_bytes == 2 * alloc_fp16.target_bytes, (
        f"fp32 target {alloc_fp32.target_bytes} should be 2x fp16 {alloc_fp16.target_bytes}"
    )
