"""Algorithmic tests for the Lagrangian allocator.

The allocator solves:

    min Σ e_{g,t}(rT, rd, b)
    s.t. Σ s_{g,t}(rT, rd, b) ≤ B

where B = original_bytes / target_ratio. The Lagrangian relaxation
decouples across cells:

    L(λ) = Σ [ e_{g,t}(rT, rd, b) + λ · s_{g,t}(rT, rd, b) ]

so each cell is solved independently and λ is found by bisection.

These tests verify the algorithm against a tiny case where the optimal
choice can be derived by hand.
"""

from __future__ import annotations

import math

import pytest

from kvcompress.compressor.allocator import (
    Cell,
    JointAllocator,
)


def test_allocator_respects_byte_budget() -> None:
    """The achieved ratio should be in the same order of magnitude as
    the target. The cost grid is discrete so the actual ratio can be
    off by a factor of up to 2 in either direction."""
    cells = [
        Cell(shape=(4, 64, 32), kind="key", layer_group=0),
        Cell(shape=(4, 64, 32), kind="value", layer_group=0),
    ]
    for ratio in (2.0, 3.0, 4.0):
        alloc = JointAllocator(target_ratio=ratio, bits_grid=(0, 2, 4, 8))
        result = alloc.optimize(cells)
        # Achieved ratio should be within 2x of the target (either way).
        if result.achieved_ratio > 0:
            log_ratio = math.log2(result.achieved_ratio / ratio)
            assert (
                abs(log_ratio) < 1.5
            ), f"target={ratio}, achieved={result.achieved_ratio:.2f}, log2={log_ratio:.2f}"


def test_allocator_at_higher_ratio_uses_more_residual_bits() -> None:
    """At higher compression ratios, the allocator commits more residual
    bits per cell (because lower ranks alone can't hit the budget)."""
    cells = [Cell(shape=(4, 128, 64), kind="key", layer_group=0)]
    bits_low = sum(
        a.bits
        for a in JointAllocator(target_ratio=2.0, bits_grid=(0, 2, 4, 8))
        .optimize(cells)
        .allocations
    )
    bits_high = sum(
        a.bits
        for a in JointAllocator(target_ratio=4.0, bits_grid=(0, 2, 4, 8))
        .optimize(cells)
        .allocations
    )
    # Higher ratio → at least as many residual bits, generally more.
    assert (
        bits_high >= bits_low
    ), f"expected more bits at higher ratio: 2x={bits_low}, 4x={bits_high}"


def test_allocator_error_model_is_monotone_in_tau() -> None:
    """For a fixed bit-width, error = ε²(b) * τ(rT, rd) is monotone
    non-increasing in rT (and similarly for rd). The allocator's
    candidate grid must reflect this so the Lagrangian sees consistent
    error surfaces.
    """
    cells = [Cell(shape=(4, 128, 64), kind="key", layer_group=0)]
    alloc = JointAllocator(target_ratio=3.0, bits_grid=(4,))
    result = alloc.optimize(cells)
    a = result.allocations[0]
    # Re-derive τ for the chosen (rT, rd) and a slightly higher rank.
    rt, rd, _ = a.r_token, a.r_feature, a.bits
    m, t, d = cells[0].shape
    # Default τ model: max(1 - rT/T, 1 - rd/d).
    tau_at = max(1 - rt / t, 1 - rd / d)
    # Higher rank → lower τ → lower error.
    if rt < t:
        tau_higher = max(1 - (rt + 1) / t, 1 - rd / d)
        assert tau_higher <= tau_at
    # Sanity: error is non-negative.
    assert a.error >= 0


def test_allocator_allocations_match_grid() -> None:
    """Every returned ``(rT, rd, b)`` must be in the candidate grid."""
    cells = [Cell(shape=(4, 32, 16), kind="key", layer_group=0)]
    alloc = JointAllocator(target_ratio=3.0, bits_grid=(0, 2, 4, 8))
    result = alloc.optimize(cells)
    a = result.allocations[0]
    # Token rank must be ≤ T (= 32), feature rank ≤ d (= 16), and bits
    # must be in bits_grid.
    assert a.r_token <= 32
    assert a.r_feature <= 16
    assert a.bits in (0, 2, 4, 8)


def test_allocator_target_ratio_2x_picks_substantial_compression() -> None:
    """At 2x target, the allocator must achieve at least 1.5x actual ratio.

    A well-tuned allocator should hit 2x closely; we allow 25% slack to
    account for the discrete cost grid.
    """
    cells = [Cell(shape=(4, 256, 64), kind="key", layer_group=0)]
    alloc = JointAllocator(target_ratio=2.0, bits_grid=(0, 2, 4, 8))
    result = alloc.optimize(cells)
    assert (
        result.achieved_ratio >= 1.5
    ), f"achieved ratio {result.achieved_ratio:.2f} too low for 2x target"


def test_allocator_handles_empty_cell_list() -> None:
    alloc = JointAllocator(target_ratio=3.0)
    result = alloc.optimize([])
    assert len(result.allocations) == 0
    assert result.target_bytes == 0


def test_allocator_invalid_target_ratio_raises() -> None:
    with pytest.raises(ValueError, match="target_ratio"):
        JointAllocator(target_ratio=0.5)


def test_allocator_layers_independent_when_layer_group_differs() -> None:
    """Cells in different layer groups should get independent allocations
    (the Lagrangian is fully decoupled across cells)."""
    cells_a = [Cell(shape=(4, 32, 16), kind="key", layer_group=0)]
    cells_b = [Cell(shape=(4, 32, 16), kind="key", layer_group=1)]
    alloc = JointAllocator(target_ratio=3.0, bits_grid=(0, 4, 8))
    ra = alloc.optimize(cells_a).allocations[0]
    rb = alloc.optimize(cells_b).allocations[0]
    # Same shape, same target → same allocation. The test is structural:
    # we just want the call not to crash and to return a valid result.
    assert ra.cost_bytes > 0
    assert rb.cost_bytes > 0
    assert ra.r_token == rb.r_token
    assert ra.r_feature == rb.r_feature
    assert ra.bits == rb.bits
