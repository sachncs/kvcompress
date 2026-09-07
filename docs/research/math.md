# Math

This document restates the key equations from the paper
[arXiv:2607.12550](https://arxiv.org/abs/2607.12550) in our notation,
and identifies where each one lives in the code.

## Notation

| Symbol | Meaning |
|---|---|
| `m` | Merged head × layer count for one cell: ``m = |g| · n_h``. |
| `T` | Token axis length (context). |
| `dh` | Per-head feature dimension. |
| `rT`, `rd` | Token / feature ranks after Tucker truncation. |
| `b` | Residual bit-width (0, 2, 4, or 8). |
| `B` | Total byte budget for one model. |
| `c` | Bytes per stored scalar (fp16 ⇒ 2). |
| `X ∈ R^{m × T × dh}` | Cell's KV tensor. |
| `X̂` | Partial Tucker approximation. |
| `R = X - X̂` | Truncation residual. |
| `Π` | JL rotation matrix (square, `dh × dh`). |
| `τ(rT, rd)` | Relative Frobenius mass discarded by partial Tucker truncation. |
| `ε²(b)` | Fraction of `τ` the residual *fails* to recover. |

## Eq. 1 — Cost

The bytes consumed by a single cell at choice `(rT, rd, b)`:

```
s(rT, rd, b) = (m·rT·rd + T·rT + dh·rd) · c + (b/8) · m · T · dh
             = tucker_core + tucker_factors + packed_residual
```

Code: `compressor/allocator.py:JointAllocator._build_cell_grid` (the
`cost = tucker_bytes + residual_bytes` line).

## Eq. 2 — Error model

```
e(rT, rd, b) ≈ ε²(b) · τ(rT, rd)
```

`τ(rT, rd)` is the relative Frobenius mass that the partial Tucker
truncation discards. `ε²(b)` is the fraction of that mass the residual
fails to recover; calibrated once on a Gaussian round-trip:
`ε²(0) = 1`, decreasing in `b`.

Code: `_DEFAULT_EPSILON_SQUARED = {0: 1.0, 2: 0.30, 4: 0.10, 8: 0.04}` in
`compressor/allocator.py`. A precomputed `tau_table` can be passed to
`JointAllocator.optimize(tau_table=...)` for empirical values.

## Eq. 3 — Global optimization

```
min Σ_{g, t} e_{g,t}(rT, rd, b)
s.t. Σ_{g, t} s_{g,t}(rT, rd, b) ≤ B
```

Code: `compressor/allocator.py:JointAllocator.optimize`.

## Eq. 4 — Lagrangian relaxation

```
L(λ) = Σ_{g, t} [ e_{g,t}(rT, rd, b) + λ · s_{g,t}(rT, rd, b) ]
```

The relaxation decouples across cells. For fixed `λ`, each cell is solved
independently by enumerating its `(rT, rd, b)` grid. `λ` is found by
bisection to drive the total cost to `B`.

Code: `compressor/allocator.py:JointAllocator._argmin_per_cell`.

## ST-HOSVD

We compute the partial Tucker decomposition by truncating each non-head
mode's leading singular subspace. The mode-k unfolding of a 3-D tensor
arranges mode-k fibres as rows; the SVD of the unfolding gives the basis
for mode k. We compute the feature mode first (smaller dimension), then
the token mode, which keeps the second-mode SVD cost bounded by `T·rd`
rather than `T·m·d`.

Code: `compressor/tucker.py:partial_tucker_st_hosvd`.

## JL projection

The JL matrix `Π ∈ R^{dh × dh}` is built with entries from N(0, 1/dh)
(Gaussian) or ±1/sqrt(dh) (Rademacher). The square form preserves the
embedding's norms in expectation. Decoding applies the inverse rotation
`Π` (cheap since it's a single matmul, not a true inverse).

Code: `compressor/jl.py`.

## Tail-mass accounting (FlashJoLT)

The randomized SVD caps the sketch at `q_cap`. The discarded mass is
estimated as `1 - sum(s_r²) / ||A||²` and reported as `tail_mass` on
the `SVDResult`. The allocator sees the *true* truncated-mass signal, so
the allocation stays accurate even when the randomized SVD under-truncates.

Code: `compressor/svd.py:SVD.randomise`, `compressor/flashjolt.py`.

## Lagrangian numerical stability

The cost grid is discrete; small changes in `λ` can produce large jumps
in the chosen `(rT, rd, b)`. We select the `λ` whose *achieved ratio* is
closest to the *target ratio* in log-space (not absolute-byte space),
because that's the metric users care about. See `JointAllocator.optimize`.