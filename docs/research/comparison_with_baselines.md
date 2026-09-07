# Comparison with baselines

How JoLT compares to the methods in the paper's Table 2 and to the
broader landscape of KV cache compression.

## Paper Table 2 (relative Frobenius error at 2×, T=1024)

| Method | Mistral K | Mistral V | LLaMA K | LLaMA V |
|---|---|---|---|---|
| JoLT | 0.009 | 0.006 | 0.009 | 0.006 |
| int4 per-token | 0.080 | 0.131 | 0.077 | 0.123 |
| xKV (cross-layer SVD) | 0.077 | 0.237 | 0.087 | 0.224 |

JoLT is roughly an order of magnitude better on both K and V than
either pure quantization or pure low-rank. The gap is largest on values
because their spectrum is nearly flat (spectral_motivation.md).

## Broader landscape

### Palu (arXiv:2407.21118)

Palu compresses the KV cache by low-rank factorizing the *projection
matrices* `W_K` and `W_V`, then storing a low-rank latent cache. JoLT
operates on the materialized K/V *after* projection. The two are
complementary: Palu reduces compute (no need to materialize full K/V),
while JoLT reduces memory (the materialized cache is compressed).

### xKV (arXiv:2503.18893)

xKV factors stacked cross-layer feature blocks. It sees the cache as a
single matrix across layers, then projects to a shared low-rank
subspace. xKV's cross-layer view is powerful when layer spectra are
similar, but the paper shows it loses to JoLT at every ratio on both
architectures (Appendix B.2).

### KIVI (arXiv:2402.02750)

KIVI quantizes the cache asymmetrically: per-token 4-bit on K, per-channel
4-bit on V. It's a pure quantization method without a low-rank backbone.
JoLT achieves ~10× lower reconstruction error at the same nominal
compression because the Tucker backbone absorbs the redundancy before
quantization.

### TurboQuant (arXiv:2504.19874)

TurboQuant is a near-optimal-distortion vector quantizer. It's
near-optimal at its native bit-rate but doesn't expose continuous ratios
in the 2-3× free zone (4-bit is its smallest fixed setting, which is
already 4×). JoLT covers the free zone continuously.

### H2O / SnapKV / StreamingLLM (eviction)

These *evict* low-importance tokens rather than compressing. The
mechanism is orthogonal to JoLT — you can run eviction first and then
apply JoLT to the surviving tokens.

## When to use which

- **JoLT / FlashJoLT:** default for any decoder-only LLM at 2-3×.
- **KIVI / TurboQuant:** when you want a fixed bit-width and don't care
  about the 2-3× free zone.
- **Palu:** when you're training / fine-tuning the projection matrices
  and want to bake compression into the model.
- **xKV:** when you have many layers and the cross-layer structure is
  exploitable (typically 10+ layers).
- **Eviction:** when the cache growth is dominated by low-importance
  tokens (chat, retrieval).
- **Identity:** when you're benchmarking and want a "no compression"
  baseline.