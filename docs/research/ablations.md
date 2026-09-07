# Ablations

The paper's Section 7 isolates the design choices behind the free zone.

## A1: allocation strategy

| Strategy | 2× | 4× | 8× |
|---|---|---|---|
| Joint (JoLT) | 6.54 | 6.93 | 8.14 |
| Greedy | 6.54 | 6.72 | 8.42 |
| Rank-only (no residual) | 7.27 | 7.95 | 8.65 |

**Reading:**

- Removing the residual entirely costs 0.4-1.0 PPL across the grid. The
  residual is what makes the backbone near-lossless.
- The joint solver is within noise of greedy at 2× but pulls ahead at
  high compression — exactly where the freedom to move budget between
  K and V matters most.

In our code: `compressor/allocator.py:JointAllocator` vs `GreedyAllocator`.

## A5: per-group vs uniform rank

| Strategy | 2× | 4× | 8× |
|---|---|---|---|
| Joint (per-group) | 6.54 | 6.93 | 8.14 |
| Uniform rank | 6.70 | 7.47 | 8.68 |

Per-group wins at every cell by up to ~1 PPL. Different layer groups
genuinely want different ranks.

In our code: `compressor/allocator.py:Cell` carries `layer_group`. Set
`layer_groups=G` in `enable_compression` to use G groups.

## Bit grid

| Residual bits | PPL @ 2× |
|---|---|
| 0 (no residual) | 7.27 |
| 2 | 6.63 |
| 4 | 6.537 |
| 8 | 6.536 |

Quality saturates at 4 bits: 4 → 8 buys nothing measurable. An 8-bit
residual also can't fit the budget above 2×, so 4 bits is the quality
knee *and* the only feasible default above 2×.

In our code: `compressor/allocator.py:bits_grid` controls which bit-widths
the allocator considers. Default `(0, 2, 4, 8)` matches the paper.

## Tail-mass accounting (FlashJoLT)

In the free zone the fast and exact backbones agree to within
`|Δ_K|, |Δ_V| ≤ 0.003` (paper Appendix D). Outside the free zone the
parity breaks because the cap policy clamps `q_cap` to 32 even when the
true rank is higher. The paper's calibration gate
(`Δ_K ≤ 0.025, speedup ≥ 3×`) is satisfied for `R ≤ 5` at every
context from 512 to 8192.

In our code: `compressor/flashjolt.py:flashjolt_cap`.

## Reproducing locally

`scripts/run_table2_reconstruction.py` reproduces the reconstruction
fidelity of Table 2 on synthetic K/V. The perplexity grid requires a
7B/13B model and is documented in `reproduction_notes.md`.

We don't ship ablations A1-A5 as standalone scripts (they require
running a full LLM with multiple configurations), but the allocator and
compressor expose everything needed:

```python
# Greedy ablation:
from kvcompress.compressor.allocator import GreedyAllocator
g = GreedyAllocator(target_ratio=3.0)
result = g.optimize(cells)

# Rank-only ablation (bits forced to 0):
from kvcompress import JoLTCompressor
comp = JoLTCompressor(compression_ratio=3.0, bits=(0,))
```