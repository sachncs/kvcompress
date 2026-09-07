# Reproduction notes

This document is about how the paper's numbers were obtained, what we
can and can't reproduce in this repo, and how to interpret any
discrepancies.

## What we can reproduce locally

- **Reconstruction fidelity (paper Table 2).** We run a synthetic
  benchmark (`scripts/run_table2_reconstruction.py`) on K/V tensors
  whose spectra mimic the paper's measured Mistral-7B layer 15 spectra.
  The numbers are qualitative, not exact — the synthetic tensors are
  not the paper's real KV cache.

- **Algorithm correctness.** Unit tests cover every math invariant the
  paper relies on:
  - Tucker partial reconstruction at full rank equals the original
    (`tests/unit/tucker_test.py::test_full_rank_is_identity`).
  - Tucker reconstruction is monotone in rank
    (`tests/unit/tucker_test.py::test_partial_tucker_reconstruction_improves_with_rank`).
  - Tail mass is monotone in rank and zero at full rank
    (`tests/unit/tucker_test.py::test_token_rank_caps_at_t`).
  - Randomized SVD with oversampling is within slack of exact
    (`tests/unit/svd_test.py::test_randomised_reconstruction_quality`).
  - Lagrangian allocator hits the byte budget to within log-space
    accuracy (`tests/unit/allocator_test.py::test_joint_allocator_hits_budget_3x`).

- **End-to-end HF integration.** `tests/integration/test_gpt2.py` runs
  GPT-2 end-to-end with compression enabled. We verify:
  - `identity` matches the baseline output exactly.
  - `flashjolt` produces output that does not crash and uses memory
    less than the baseline.
  - `disable()` restores baseline output exactly.

## What we cannot reproduce locally

- **Mistral-7B perplexity (paper Table 1).** Requires loading Mistral-7B
  weights and running WikiText-2 evaluation. The repo's CI does not have
  GPU resources for this; we leave the run script as a guide but do not
  assert numerical parity.
- **LLaMA-2-13B perplexity.** Same as above, larger model.
- **GSM8K accuracy.** Same as above; requires a reasoning-eval harness.
- **RULER needle-in-haystack.** Same; we ship a simplified
  single-needle script for smoke testing but no multi-needle sweep.

To run any of these locally, you need:

- ≥40 GB GPU (A100-40GB or H100).
- The corresponding model weights (`mistralai/Mistral-7B-v0.3`,
  `meta-llama/Llama-2-13b-hf`).
- `pip install "kvcompress[bench]"` for datasets and pandas.

## Synthetic benchmark caveats

Our synthetic KV generator uses singular-value-like decay (`1/i`) on the
core's components to match the paper's measured decay. The exact decay
slopes differ from real Mistral-7B layer 15, so the *quantitative*
reconstruction errors we report differ from the paper. The qualitative
ordering (JoLT ≪ int4 ≪ lowrank on reconstruction fidelity) is
preserved.

## Calibration table

The allocator uses `ε²(b) = {0: 1.0, 2: 0.30, 4: 0.10, 8: 0.04}`. These
are illustrative numbers calibrated on a Gaussian round-trip; the paper
doesn't publish the exact values. To improve, pass `tau_table` and
`epsilon_squared` overrides to `JointAllocator`:

```python
from kvcompress.compressor.allocator import JointAllocator

alloc = JointAllocator(
    target_ratio=3.0,
    epsilon_squared={0: 1.0, 2: 0.30, 4: 0.10, 8: 0.04},
    # tau_table=...  # optional: empirical tail masses from your model
)
```

## What changed between the paper and our implementation

| Paper | Ours |
|---|---|
| Lagrangian bisection on `λ` with monotonic grid | Dense log-space scan + bracketed bisection. Discrete cost grid has jumps, so we select the `λ` whose *achieved ratio* is closest to the target in log space, not absolute byte distance. |
| ε² calibrated on real Mistral data | Default `ε²` is a Gaussian round-trip calibration. Override per-model for paper parity. |
| Tail-mass accounting for FlashJoLT | Reported via `SVDResult.tail_mass`; the allocator sees the true truncated mass. |
| Per-layer τ table from calibration | Not shipped by default; `tau_table` parameter accepts one if you compute it. |

These differences are documented in
[`docs/research/math.md`](math.md) and the code comments.

## Open questions

1. The paper's reconstruction parity between FlashJoLT and exact JoLT
   holds at `|Δ| ≤ 0.003` in the free zone. We verify this for our
   small-T cases; we don't have the compute to verify for T=8K.
2. The Lagrangian multiplier's convergence on the paper's `B` budget is
   exact (3 decimals) in the free zone. Ours is within log-space
   accuracy due to the discrete grid.
3. The paper uses LBFGS / custom schedules for the Lagrangian; we use
   bisection. The two are mathematically equivalent for the
   relaxed-but-discretized problem.