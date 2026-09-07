# Free zone

The paper's central empirical claim is a **near-lossless 2-3× free zone**
that holds across perplexity, GSM8K accuracy, and RULER needle retrieval
on both a GQA model (Mistral-7B-v0.3) and an MHA model (LLaMA-2-13B).

## Definition

For compression ratios `R ∈ [2, 3]`, the *drift* between compressed and
uncompressed performance is at or within statistical noise. Concretely:

| | Mistral-7B | LLaMA-2-13B |
|---|---|---|
| Baseline PPL (T=1024) | 6.28 | 5.39 |
| PPL @ 2× | 6.28 (-0.02%) | 5.39 (-0.01%) |
| PPL @ 3× | 6.29 (+0.04%) | 5.39 (-0.01%) |
| PPL @ 4× | 6.64 (+5.58%) | 6.86 (+27.28%) |
| PPL @ 5× | 7.01 (+11.56%) | 9.07 (+68.30%) |

Both architectures are noise-flat at 2-3×. Beyond, they diverge sharply:
Mistral degrades gracefully, LLaMA breaks.

## Mechanism

Two factors together produce the free zone:

1. **Reconstruction fidelity.** At 2×, JoLT reaches `K=0.009`, `V=0.006`
   relative Frobenius error (paper Table 2). That's about one bin's
   worth of fp16 noise.
2. **Attention insensitivity.** The softmax in attention is robust to
   small perturbations in K and V — especially in early layers where the
   residual stream dominates.

The combination means a small-but-nonzero cache error has essentially
zero downstream effect on the model's outputs.

## Free-zone boundaries

The free zone ends where reconstruction error grows large enough that
attention starts to "feel" the corruption. Two thresholds:

- **Quantization-bound (all architectures):** at fixed bit-width `b`,
  the residual can't recover more than `1 - ε²(b)` of the discarded
  mass. Once `τ` (truncation tail) is bigger than that, error stops
  decreasing with `rT`.
- **Architecture-bound (MHA only):** LLaMA's value spectrum is flat
  (paper Section 3), so the residual hits the quantization-bound earlier.
  At 4× and beyond, the residual can't keep up, and reconstruction
  error climbs rapidly. Mistral's value spectrum is *also* flat but
  Mistral has fewer KV heads (GQA), so the per-head rank budget is
  larger and the residual is more effective.

## Practical implications

- **Default to 3× for any production deployment.** You're in the free
  zone regardless of architecture.
- **5× is the next stop for GQA models.** Beyond 5×, even GQA starts to
  degrade meaningfully on reasoning tasks (GSM8K).
- **Stay at 3× for MHA at long context.** The 4-5× jump that GQA
  tolerates is where MHA breaks.

## How we test the free zone

`scripts/run_table2_reconstruction.py` reproduces the reconstruction
fidelity numbers (Table 2 of the paper). The perplexity grid
(paper Table 1) requires a 7B+ model and is documented in
`reproduction_notes.md`.