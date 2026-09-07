# Paper notes

> **Authorship.** sachin is the implementer of this library, not the
> paper's author. The JoLT algorithm is from
> *Krishnan, R. & Schulz, V. (2026). "A JoLT for the KV Cache: Near-Lossless
> KV Cache Compression via Joint Tucker and JL-Residual Allocation for
> LLMs." arXiv:2607.12550.* This document is sachin's reading notes on
> the paper, not a summary written by the paper's authors.

Caveats and limitations of the paper that this repo inherits.

## Edge cases the paper does not cover

1. **Encoder-decoder models.** The paper studies decoder-only LLMs
   (Mistral, LLaMA). JoLT works on encoder-decoders at the cache
   level but the allocator's cost model assumes a single (m, T, dh)
   layout per cell; cross-attention KV caches don't fit that pattern.
2. **Multi-modal models.** Same as encoder-decoder — the cross-modal KV
   caches have layout the allocator doesn't model.
3. **Sliding-window attention.** Mistral uses sliding-window; the
   paper handles this implicitly because the cache is windowed by HF
   before we see it. Our HF adapter does not currently patch windowing
   behaviour; if you have a Mistral model with a custom window, the
   DynamicCache subclass will still work.
4. **State-space models.** Mamba and friends don't have a KV cache.
5. **Multi-query attention with n_kv=1.** The head axis becomes a
   singleton; partial Tucker still works but the speedup vs. plain
   low-rank is smaller.

## Caveats stated in the paper itself

From the conclusion:

> Two further caveats are worth stating directly. The compressed LLaMA
> RULER setting does not fit in 40GB because of the transient
> reconstruction copy and was run on larger hardware, a cost the same
> fused kernel would remove; and our byte accounting uses an fp16
> serialization convention for the low-rank factors and idealized
> bit-packed bytes for the quantization baselines, so rankings within a
> class are exact while cross-class achieved ratios are convention-based.

Implications for this repo:

- The 128K-context LLaMA experiment needs more GPU memory than a single
  A100-40GB. Our CI cannot reproduce.
- Cross-class (JoLT vs int4) byte counts use slightly different
  conventions. Within a class, the accounting is exact.

## Compute limits

> Finally, this study was conducted under a single-GPU (A100-40GB)
> compute budget; broader sweeps, larger models, and the harder
> multi-needle long-context regime are left to future work with more
> compute.

The repo's CI uses CPU torch and cannot reach these numbers. Tests that
require GPU resources are marked `@pytest.mark.gpu` and excluded from
the default CI run.

## What we tested vs what the paper tested

| Claim | Paper | This repo |
|---|---|---|
| Free-zone perplexity on Mistral-7B | yes | no (model too large) |
| Free-zone perplexity on LLaMA-2-13B | yes | no (model too large) |
| Reconstruction fidelity at 2× on Mistral | yes (real KV) | yes (synthetic, qualitative) |
| Reconstruction fidelity vs int4 baseline | yes | yes |
| Reconstruction fidelity vs xKV baseline | yes | no (xKV not implemented) |
| GSM8K degradation past the free zone | yes | no |
| RULER single-needle at 8K-16K | yes | partial (GPT-2 smoke test) |
| RULER multi-needle | yes | no |
| FlashJoLT speedup 5-13× | yes | no (we don't have GPU) |
| Free-zone parity exact vs FlashJoLT | yes (`|Δ| ≤ 0.003`) | partial (qualitative) |

## Recommended reading order

1. [Concepts](../user/concepts.md) — vocabulary.
2. [Quickstart](../quickstart.md) — try it.
3. [Math](math.md) — the formal algorithm.
4. [Algorithm walkthrough](algorithm.md) — code ↔ math.
5. [Spectral motivation](spectral_motivation.md) — why this works.
6. [Free zone](free_zone.md) — when it works.
7. [Comparison with baselines](comparison_with_baselines.md) — vs other methods.
8. [Reproduction notes](reproduction_notes.md) — what we did and didn't reproduce.
## vLLM integration

Two integration shapes ship with the library today:

### Shape A — `kvcompress.adapters.vllm.export_kv` / `import_kv`

User-driven workflow that works on any HF or vLLM-style model with a
`DynamicCache`:

```python
from vllm import LLM
from kvcompress.adapters.vllm import export_kv

llm = LLM(model="meta-llama/Llama-2-7b-hf")
llm.generate(["Hello, my name is"])
export_kv(llm, "kv.safetensors", method="flashjolt", compression_ratio=3.0)
```

The exported file is a single safetensors with one tensor per
(layer, kind) cell plus a `.meta.json` sidecar. `import_kv` does the
reverse.

### Shape B — `kvcompress.adapters.vllm_kv_offload.JoLTOffloadWorker`

Subclasses `vllm.v1.kv_offload.base.KVCacheOffloadWorker` so vLLM's
block-eviction path uses our compressor. Requires a real vLLM + CUDA
install; this Mac (and CI) can structurally validate the subclass but
cannot integration-test the runtime path.

Three integration levels, ranked by invasiveness:

1. **Shape A (current, useful):** export/import helpers. Works today on
   any model with a `DynamicCache`. Doesn't touch vLLM's scheduler.
2. **Shape B (current, structural):** `KVCacheOffloadWorker` subclass.
   Plugs into vLLM v1's offload path. Needs GPU validation.
3. **Shape C (future):** custom attention backend with per-block
   compression in the attention kernel itself. Requires deep vLLM
   expertise + CUDA work.

The right path for high-throughput serving is Shape C, but it requires
dedicated vLLM expertise. Shape A and B together cover the realistic
near-term needs.
