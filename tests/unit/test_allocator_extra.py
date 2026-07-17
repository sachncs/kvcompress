"""Coverage tests for allocator.py edge paths and dataclass methods."""

from __future__ import annotations

import pytest

from kvcompress.compressor.allocator import (
    Allocation,
    AllocationResult,
    Cell,
    GreedyAllocator,
    JointAllocator,
    bytes_original,
    candidate_feature_ranks,
    candidate_token_ranks,
)


def test_allocation_tuple_property() -> None:
    """Allocation.tuple returns (r_token, r_feature, bits)."""
    a = Allocation(r_token=8, r_feature=4, bits=4, cost_bytes=128, error=0.1)
    assert a.tuple == (8, 4, 4)


def test_allocation_result_iteration() -> None:
    """AllocationResult is iterable over its allocations."""
    r = AllocationResult(
        allocations=[
            Allocation(r_token=1, r_feature=1, bits=0, cost_bytes=10, error=0.0),
            Allocation(r_token=2, r_feature=2, bits=0, cost_bytes=20, error=0.0),
        ],
    )
    assert len(r) == 2
    assert list(r) == r.allocations


def test_allocation_result_zero_target_bytes() -> None:
    """achieved_ratio collapses to 1.0 when total_bytes is zero."""
    r = AllocationResult()
    assert r.achieved_ratio == 1.0
    assert r.total_bytes == 0


def test_bytes_original_element_size_default() -> None:
    # (2, 4, 8) = 64 elements * 2 bytes (default fp16) = 128.
    assert bytes_original((2, 4, 8)) == 128


def test_bytes_original_with_custom_element_size() -> None:
    # 64 elements * 4 bytes (fp32) = 256.
    assert bytes_original((2, 4, 8), element_size_bytes=4) == 256


def test_candidate_feature_ranks_small() -> None:
    """For small d, candidates enumerate all ranks 1..d."""
    ranks = candidate_feature_ranks(d=4)
    assert ranks == [1, 2, 3, 4]


def test_candidate_feature_ranks_large_subset() -> None:
    """For d > 32, candidates use a logarithmic-spaced subset."""
    ranks = candidate_feature_ranks(d=128)
    assert 1 in ranks
    assert 128 in ranks
    assert max(ranks) <= 128


def test_candidate_token_ranks_small() -> None:
    ranks = candidate_token_ranks(t_max=8)
    assert ranks == list(range(1, 9))


def test_candidate_token_ranks_large_subset() -> None:
    ranks = candidate_token_ranks(t_max=512)
    assert 1 in ranks
    assert 512 in ranks
    assert max(ranks) <= 512


def test_joint_allocator_invalid_target_ratio() -> None:
    """target_ratio <= 1 raises ValueError at construction."""
    with pytest.raises(ValueError, match="must be > 1.0"):
        JointAllocator(target_ratio=1.0)


def test_joint_allocator_empty_cells() -> None:
    """optimize with no cells returns an empty AllocationResult."""
    alloc = JointAllocator(target_ratio=3.0)
    result = alloc.optimize([])
    assert result.allocations == []
    assert result.total_bytes == 0
    assert result.target_bytes == 0


def test_joint_allocator_with_explicit_target_bytes() -> None:
    """Passing ``original_bytes`` overrides the per-cell sum."""
    alloc = JointAllocator(target_ratio=2.0)
    cells = [Cell(shape=(2, 8, 4), kind="key")]
    result = alloc.optimize(cells, original_bytes=1000)
    # target_bytes = 1000 / 2 = 500
    assert result.target_bytes == 500


def test_joint_allocator_candidate_token_ranks_override() -> None:
    alloc = JointAllocator(target_ratio=2.0, max_token_rank=8)
    cell = Cell(
        shape=(2, 64, 4),
        kind="key",
        candidate_token_ranks=(2, 4, 8),
    )
    result = alloc.optimize([cell])
    assert result.allocations
    # The chosen r_token must be from the override set.
    assert result.allocations[0].r_token in {2, 4, 8}


def test_joint_allocator_uses_tau_table() -> None:
    """When tau_table is provided, the optimizer uses it instead of the default model."""
    alloc = JointAllocator(target_ratio=2.0)
    cell = Cell(shape=(2, 8, 4), kind="key")
    # tau_table is keyed by (cell idx, rt * d + rd). d=4 here.
    # Provide enough rows so rank-2 captures everything (error=0).
    tau = [1.0] * (8 * 4)
    tau[2 * 4 + 2] = 0.0  # rt=2, rd=2
    tau_table = {0: tau}
    result = alloc.optimize([cell], tau_table=tau_table)
    # The picked allocation has error 0 when (rT, rF) == (2, 2).
    best = min(result.allocations, key=lambda a: a.error)
    assert best.r_token == 2 and best.r_feature == 2
    assert best.error == 0.0


def test_greedy_allocator_runs() -> None:
    """Greedy allocator covers its allocation path."""
    alloc = GreedyAllocator(target_ratio=2.0)
    cells = [Cell(shape=(2, 8, 4), kind="key")]
    result = alloc.optimize(cells)
    assert result.allocations
    assert result.target_ratio == 2.0


def test_greedy_allocator_empty_cells() -> None:
    alloc = GreedyAllocator(target_ratio=2.0)
    result = alloc.optimize([])
    assert result.allocations == []


def test_greedy_allocator_invalid_ratio() -> None:
    # GreedyAllocator doesn't validate at construction (the value is
    # only used during optimize); assert that construction succeeds.
    alloc = GreedyAllocator(target_ratio=0.5)
    assert alloc.target_ratio == 0.5


def test_joint_allocator_target_bytes_from_sum() -> None:
    """When original_bytes is not provided, sum cell bytes."""
    alloc = JointAllocator(target_ratio=2.0)
    cells = [Cell(shape=(2, 8, 4), kind="key"), Cell(shape=(2, 8, 4), kind="value")]
    result = alloc.optimize(cells)
    expected = bytes_original((2, 8, 4)) + bytes_original((2, 8, 4))
    assert result.target_bytes == expected // 2
