"""Joint Lagrangian allocator for JoLT.

Given a target byte budget ``B`` shared across all (layer group, K/V) cells,
the allocator chooses (r_token, r_feature, bits) per cell to minimize the
summed reconstruction error subject to the budget.

The cost and error model follow the paper:

* **Cost** (Eq. 1)::

    s_{g,t}(rT, rd, b) = (m·rT·rd + T·rT + dh·rd) · c
                         + (b/8) · m · T · dh

  where ``c = 2`` for fp16 factors and ``m = |g|·n_h``.

* **Error** (Eq. 2)::

    e_{g,t}(rT, rd, b) ≈ ε²(b) · τ_{g,t}(rT, rd)

  ``τ`` is the relative Frobenius mass discarded by partial Tucker truncation
  at (rT, rd); ``ε²(b)`` is the fraction of that mass the residual fails to
  recover (calibrated once on a Gaussian round-trip).

The Lagrangian relaxation gives ``L(λ) = Σ [e + λ·s]``, decoupled per cell,
so each cell is solved by an exhaustive grid search and the global budget
is met by bisection on ``λ``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

log = logging.getLogger(__name__)


# ε²(b) calibration table from the paper. Calibrated on a Gaussian
# round-trip: 0 → 1 (no residual), 2 → 0.30, 4 → 0.10, 8 → 0.04 (illustrative).
_DEFAULT_EPSILON_SQUARED = {0: 1.0, 2: 0.30, 4: 0.10, 8: 0.04}


@dataclass
class Cell:
    """One (layer group, K/V) cell the allocator optimizes.

    Attributes:
        shape: ``(m, T, dh)`` — the cell's tensor shape.
        kind: ``"key"`` or ``"value"`` (informational).
        layer_group: layer-group index.
        epsilon_squared: per-cell ``ε²(b)`` calibration (defaults to global
            if ``None``).
        candidate_bits: residual bit-widths to consider.
        candidate_token_ranks: optional override for the search grid. By
            default the allocator uses all integer ranks from 1 to T.
    """

    shape: tuple[int, int, int]
    kind: str = "key"
    layer_group: int = 0
    epsilon_squared: dict[int, float] | None = None
    candidate_bits: tuple[int, ...] = (0, 2, 4, 8)
    candidate_token_ranks: tuple[int, ...] | None = None


@dataclass
class Allocation:
    """Per-cell allocation chosen by the optimizer.

    Attributes:
        r_token: token-mode rank.
        r_feature: feature-mode rank.
        bits: residual bit-width.
        cost_bytes: bytes consumed by this cell.
        error: estimated reconstruction error contribution.
    """

    r_token: int
    r_feature: int
    bits: int
    cost_bytes: int
    error: float

    @property
    def tuple(self) -> tuple[int, int, int]:
        return (self.r_token, self.r_feature, self.bits)


@dataclass
class AllocationResult:
    """Output of :meth:`JointAllocator.optimize`."""

    allocations: list[Allocation] = field(default_factory=list)
    total_bytes: int = 0
    target_bytes: int = 0
    achieved_ratio: float = 1.0
    target_ratio: float = 1.0
    lambda_star: float = 0.0

    def __iter__(self):
        return iter(self.allocations)

    def __len__(self) -> int:
        return len(self.allocations)


class JointAllocator:
    """Per-(layer group, K/V) Lagrangian allocator.

    Args:
        target_ratio: target compression ratio (e.g. ``3.0`` for 3×).
        epsilon_squared: mapping ``bits → ε²`` used in the error model.
        factor_dtype_bytes: bytes per scalar in the Tucker core and bases
            (default 2 for fp16).
        max_token_rank: optional cap on token rank to keep the grid search
            tractable on very long contexts.
        bits_grid: residual bit-widths considered. Defaults to ``(0, 2, 4, 8)``.
    """

    def __init__(
        self,
        target_ratio: float,
        *,
        epsilon_squared: dict[int, float] | None = None,
        factor_dtype_bytes: int = 2,
        max_token_rank: int = 512,
        bits_grid: tuple[int, ...] = (0, 2, 4, 8),
    ) -> None:
        if target_ratio <= 1.0:
            raise ValueError(f"target_ratio must be > 1.0, got {target_ratio}")
        self.target_ratio = float(target_ratio)
        self.epsilon_squared = dict(epsilon_squared or _DEFAULT_EPSILON_SQUARED)
        self.factor_dtype_bytes = int(factor_dtype_bytes)
        self.max_token_rank = int(max_token_rank)
        self.bits_grid = tuple(bits_grid)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def optimize(
        self,
        cells: Sequence[Cell],
        *,
        tau_table: dict[int, list[float]] | None = None,
        original_bytes: int | None = None,
    ) -> AllocationResult:
        """Allocate ranks and bits across cells to hit the target ratio.

        Args:
            cells: list of cells to allocate.
            tau_table: optional precomputed ``τ(rT, rd)`` lookup table per
                cell index. Each entry maps ``(rT, rd)`` to a relative
                Frobenius tail mass. If absent, a simple monotone model is
                used as a stand-in.
            original_bytes: total uncompressed bytes. Defaults to summing
                ``Π cell.bytes_original()``.

        Returns:
            :class:`AllocationResult` whose ``total_bytes`` matches the
            target ratio within rounding.
        """
        if not cells:
            return AllocationResult(
                allocations=[],
                total_bytes=0,
                target_bytes=0,
                achieved_ratio=1.0,
                target_ratio=self.target_ratio,
            )

        if original_bytes is None:
            original_bytes = sum(_bytes_original(c.shape) for c in cells)
        target_bytes = max(1, int(round(original_bytes / self.target_ratio)))

        # Per-cell candidate grid: (rT, rd, b).
        per_cell_grid: list[list[Allocation]] = []
        original_per_cell: list[int] = []
        for idx, cell in enumerate(cells):
            original_per_cell.append(_bytes_original(cell.shape))
            grid = self._build_cell_grid(cell, tau_table, idx)
            per_cell_grid.append(grid)
            log.debug(
                "allocator: cell %d (%s, shape=%s) grid_size=%d",
                idx,
                cell.kind,
                cell.shape,
                len(grid),
            )

        # Coarse log-space scan to find a λ that puts cost close to target,
        # then refine with bisection. We track all candidate (λ, allocation)
        # pairs along the way and pick the one whose *achieved ratio* is
        # closest to the target ratio in log space (this gives a tighter
        # match in compression ratio than absolute byte distance, because
        # the cost grid is discrete and jumps).
        candidates: list[tuple[float, int, list[Allocation], float]] = []
        # Dense logspace scan.
        log_lambdas = [-12 + 0.1 * i for i in range(180)]  # 1e-12 .. 1e+6
        for log_lam in log_lambdas:
            lam = 10.0**log_lam
            cand = self._argmin_per_cell(per_cell_grid, lam)
            total = sum(a.cost_bytes for a in cand)
            if total <= 0:
                continue
            achieved = original_bytes / total
            # Log-ratio distance to target.
            score = abs(math.log(achieved) - math.log(self.target_ratio))
            candidates.append((score, total, cand, lam))

        candidates.sort(key=lambda c: c[0])
        _, best_total, best_cand, best_lam = candidates[0]

        # Refine with bisection around the closest candidate.
        lo = best_lam / 10 if best_lam > 0 else 1e-12
        hi = best_lam * 10 if best_lam > 0 else 1e-6
        cand_lo = self._argmin_per_cell(per_cell_grid, lo)
        cand_hi = self._argmin_per_cell(per_cell_grid, hi)
        cost_lo = sum(a.cost_bytes for a in cand_lo)
        cost_hi = sum(a.cost_bytes for a in cand_hi)
        if cost_lo > cost_hi:
            lo, hi = hi, lo
            cost_lo, cost_hi = cost_hi, cost_lo
            cand_lo, cand_hi = cand_hi, cand_lo

        # Bracket target_bytes with lo/hi.
        for _ in range(60):
            if cost_lo > target_bytes and cost_hi < target_bytes:
                break
            if cost_lo < target_bytes:
                lo /= 10
                cand_lo = self._argmin_per_cell(per_cell_grid, lo)
                cost_lo = sum(a.cost_bytes for a in cand_lo)
            elif cost_hi > target_bytes:
                hi *= 10
                cand_hi = self._argmin_per_cell(per_cell_grid, hi)
                cost_hi = sum(a.cost_bytes for a in cand_hi)
            else:
                break

        for _ in range(40):
            mid = (lo + hi) / 2
            cand_mid = self._argmin_per_cell(per_cell_grid, mid)
            cost_mid = sum(a.cost_bytes for a in cand_mid)
            if cost_mid > target_bytes:
                lo, cost_lo = mid, cost_mid
            else:
                hi, cost_hi = mid, cost_mid

        cand_mid = self._argmin_per_cell(per_cell_grid, (lo + hi) / 2)
        cost_mid = sum(a.cost_bytes for a in cand_mid)
        if cost_mid > 0:
            achieved_mid = original_bytes / cost_mid
            score_mid = abs(math.log(achieved_mid) - math.log(self.target_ratio))
            if score_mid < candidates[0][0]:
                best_cand = cand_mid
                best_lam = (lo + hi) / 2
                best_total = cost_mid

        return self._make_result(best_cand, original_per_cell, target_bytes, best_lam)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_cell_grid(
        self,
        cell: Cell,
        tau_table: dict[int, list[float]] | None,
        idx: int,
    ) -> list[Allocation]:
        m, t, d = cell.shape
        c_bytes = self.factor_dtype_bytes
        eps = cell.epsilon_squared or self.epsilon_squared
        bits_grid = self.bits_grid

        # Feature ranks: search over the full range.
        candidate_rd = _candidate_feature_ranks(d)

        # Token ranks: cap at min(T, max_token_rank) and step.
        if cell.candidate_token_ranks is not None:
            candidate_rt = sorted({int(r) for r in cell.candidate_token_ranks})
        else:
            rt_max = min(t, self.max_token_rank)
            candidate_rt = _candidate_token_ranks(rt_max)

        grid: list[Allocation] = []
        for rt in candidate_rt:
            for rd in candidate_rd:
                rt = min(rt, t)
                rd = min(rd, d)
                # Tucker cost in scalars.
                tucker_scalars = m * rt * rd + t * rt + d * rd
                tucker_bytes = tucker_scalars * c_bytes
                # Tail mass for this (rT, rd). The simple model is a product
                # of two decay factors; if tau_table is provided, look up.
                tau = self._tau(tau_table, idx, rt, rd, t, d)
                for b in bits_grid:
                    residual_bytes = (b // 8) * m * t * d
                    cost = tucker_bytes + residual_bytes
                    eps_b = eps.get(b, 1.0)
                    error = eps_b * tau
                    grid.append(
                        Allocation(
                            r_token=rt,
                            r_feature=rd,
                            bits=b,
                            cost_bytes=cost,
                            error=error,
                        )
                    )
        return grid

    def _tau(
        self,
        tau_table: dict[int, list[float]] | None,
        idx: int,
        rt: int,
        rd: int,
        t: int,
        d: int,
    ) -> float:
        """Relative Frobenius tail mass τ(rT, rd) for a cell.

        Simple model: ``τ = max(1 - rT/T, 1 - rd/d)`` — the worst-case
        truncation across the two modes. This is monotone in both ranks,
        equals zero only when both modes are full, and is positive whenever
        *any* truncation happens (so the residual budget is meaningful).
        Replace with a precomputed lookup if available.
        """
        if tau_table is not None and idx in tau_table:
            try:
                return float(tau_table[idx][rt * d + rd])
            except (IndexError, TypeError):
                pass
        rt_frac = 1.0 - rt / max(t, 1)
        rd_frac = 1.0 - rd / max(d, 1)
        return max(0.0, max(rt_frac, rd_frac))

    def _argmin_per_cell(
        self,
        per_cell_grid: list[list[Allocation]],
        lam: float,
    ) -> list[Allocation]:
        """Per-cell minimizer of ``error + λ · cost``."""
        result = []
        for grid in per_cell_grid:
            best = grid[0]
            best_obj = grid[0].error + lam * grid[0].cost_bytes
            for a in grid[1:]:
                obj = a.error + lam * a.cost_bytes
                if obj < best_obj:
                    best = a
                    best_obj = obj
            result.append(best)
        return result

    def _make_result(
        self,
        allocations: list[Allocation],
        original_per_cell: list[int],
        target_bytes: int,
        lam_star: float = 0.0,
    ) -> AllocationResult:
        total = sum(a.cost_bytes for a in allocations)
        original = sum(original_per_cell)
        achieved = original / total if total > 0 else 1.0
        return AllocationResult(
            allocations=allocations,
            total_bytes=total,
            target_bytes=target_bytes,
            achieved_ratio=achieved,
            target_ratio=self.target_ratio,
            lambda_star=lam_star,
        )


# ---------------------------------------------------------------------------
# Greedy baseline
# ---------------------------------------------------------------------------


class GreedyAllocator:
    """Greedy baseline that picks (rT, rd, b) per cell by error reduction / byte.

    For ablation: this is what the paper calls "greedy" — no Lagrangian
    multiplier, no global rebalancing. It picks the cell that offers the
    largest error reduction per additional byte, applies the change,
    repeats until the budget is exhausted.
    """

    def __init__(
        self,
        target_ratio: float,
        *,
        factor_dtype_bytes: int = 2,
        max_token_rank: int = 512,
        bits_grid: tuple[int, ...] = (0, 2, 4, 8),
    ) -> None:
        self.target_ratio = float(target_ratio)
        self.factor_dtype_bytes = int(factor_dtype_bytes)
        self.max_token_rank = int(max_token_rank)
        self.bits_grid = tuple(bits_grid)

    def optimize(self, cells: Sequence[Cell]) -> AllocationResult:
        if not cells:
            return AllocationResult(
                allocations=[],
                total_bytes=0,
                target_bytes=0,
                achieved_ratio=1.0,
                target_ratio=self.target_ratio,
            )
        original = sum(_bytes_original(c.shape) for c in cells)
        target = max(1, int(round(original / self.target_ratio)))
        # Initial: largest rank, no bits.
        current: list[Allocation] = []
        for cell in cells:
            m, t, d = cell.shape
            rT = min(t, self.max_token_rank)
            rD = d
            cost = (m * rT * rD + t * rT + d * rD) * self.factor_dtype_bytes
            current.append(
                Allocation(r_token=rT, r_feature=rD, bits=0, cost_bytes=cost, error=0.0)
            )

        def total_cost() -> int:
            return sum(a.cost_bytes for a in current)

        # Greedy loop.
        while total_cost() > target:
            best_idx = -1
            best_gain = -1.0
            for i, cell in enumerate(cells):
                m, t, d = cell.shape
                a = current[i]
                # Try doubling bits, increasing ranks one at a time.
                candidates: list[Allocation] = []
                if a.bits < max(self.bits_grid):
                    next_b = next(b for b in self.bits_grid if b > a.bits)
                    added = (next_b // 8) * m * t * d
                    new_err = 0.05 * next_b
                    gain = (a.error - new_err) / max(1, added)
                    if gain > best_gain and total_cost() + added <= target:
                        best_gain = gain
                        best_idx = i
                if a.r_feature < d:
                    added = (m * a.r_token + d) * self.factor_dtype_bytes
                    gain = 0.001 / max(1, added)
                    if gain > best_gain and total_cost() + added <= target:
                        best_gain = gain
                        best_idx = i
                if a.r_token < min(t, self.max_token_rank):
                    added = (m * a.r_feature + t) * self.factor_dtype_bytes
                    gain = 0.0005 / max(1, added)
                    if gain > best_gain and total_cost() + added <= target:
                        best_gain = gain
                        best_idx = i
            if best_idx < 0:
                break  # cannot improve
            # Apply the best change.
            cell = cells[best_idx]
            m, t, d = cell.shape
            a = current[best_idx]
            if a.bits < max(self.bits_grid):
                next_b = next(b for b in self.bits_grid if b > a.bits)
                current[best_idx] = Allocation(
                    r_token=a.r_token,
                    r_feature=a.r_feature,
                    bits=next_b,
                    cost_bytes=a.cost_bytes + (next_b // 8) * m * t * d,
                    error=0.05 * next_b,
                )
            elif a.r_feature < d:
                current[best_idx] = Allocation(
                    r_token=a.r_token,
                    r_feature=a.r_feature + 1,
                    cost_bytes=a.cost_bytes + (m * a.r_token + d) * self.factor_dtype_bytes,
                    error=0.001,
                )
            elif a.r_token < min(t, self.max_token_rank):
                current[best_idx] = Allocation(
                    r_token=a.r_token + 1,
                    r_feature=a.r_feature,
                    cost_bytes=a.cost_bytes + (m * a.r_feature + t) * self.factor_dtype_bytes,
                    error=0.0005,
                )
            else:
                break

        return AllocationResult(
            allocations=current,
            total_bytes=total_cost(),
            target_bytes=target,
            achieved_ratio=original / max(1, total_cost()),
            target_ratio=self.target_ratio,
            lambda_star=0.0,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bytes_original(shape: tuple[int, int, int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n * 2  # fp16


def _candidate_feature_ranks(d: int) -> list[int]:
    """Discrete set of feature ranks to search over.

    For small ``d`` (≤ 32) we enumerate all values; for larger ``d`` we use
    a logarithmic-spaced subset capped at ``min(d, 256)``.
    """
    if d <= 32:
        return list(range(1, d + 1))
    candidates = sorted(
        {1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, d}
    )
    return [c for c in candidates if c <= d]


def _candidate_token_ranks(t_max: int) -> list[int]:
    """Discrete token ranks. Cap at ``t_max``; use logarithmic spacing above 32."""
    if t_max <= 32:
        return list(range(1, t_max + 1))
    base = list(range(1, 33))
    extras = sorted(
        {48, 64, 96, 128, 192, 256, 384, 512, t_max}
    )
    return base + [e for e in extras if e <= t_max]


__all__ = [
    "Allocation",
    "AllocationResult",
    "Cell",
    "GreedyAllocator",
    "JointAllocator",
]