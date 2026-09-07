# Performance guide

This page is about squeezing the most out of `kvcompress` for your
specific workload.

## TL;DR

- **Default to 3× (target_memory="33%").** You're in the paper's
  near-lossless "free zone" for both GQA and MHA models.
- **Use `flashjolt`, not `jolt`.** 5-13× faster at matched quality.
- **Layer groups are usually unnecessary.** The paper's default is
  `layer_groups=1`; use > 1 only if you have a measured reason.
- **Residual bits default to `(0, 2, 4, 8)`.** The allocator picks
  per-cell; don't fix it unless you have a hardware reason.

## Picking a compression ratio

The paper's "free zone" is **2-3×** for both GQA and MHA models.
Within this band the model's perplexity drifts by < 1% (statistical
noise). Outside the band the architectures diverge:

| Ratio | Mistral-7B (GQA) | LLaMA-2-13B (MHA) | Recommendation |
|---|---|---|---|
| 1× (baseline) | 6.28 PPL | 5.39 PPL | No compression. |
| 2× | 6.28 PPL | 5.39 PPL | **Free zone — default.** |
| 3× | 6.29 PPL | 5.39 PPL | **Free zone — safe for production.** |
| 4× | 6.64 PPL | 6.86 PPL | GQA: OK. MHA: starts to hurt. |
| 5× | 7.01 PPL | 9.07 PPL | GQA: still OK. MHA: unusable. |
| 8× | — | — | Lossy on both. |

Source: paper Table 1 (Mistral-7B-v0.3 and LLaMA-2-13B on WikiText-2/C4 at
T=1024).

### Default for production

```python
enable_compression(model, method="flashjolt", target_memory="33%")
```

That's 3×. Both GQA and MHA models stay in the free zone.

### Aggressive for GQA-only deployments

If you've confirmed your model is GQA (Mistral, Qwen2, Gemma2), you
can push harder:

```python
enable_compression(model, method="flashjolt", target_memory="20%")  # 5×
enable_compression(model, method="flashjolt", target_memory="14%")  # 7×
```

### Stay in the free zone for MHA

If your model is MHA (LLaMA-1/2, Phi-2), stay at 3× and below:

```python
enable_compression(model, method="flashjolt", target_memory="33%")  # 3× max
```

## FlashJoLT vs JoLT

| | JoLT (exact) | FlashJoLT (randomised) |
|---|---|---|
| Compression-time SVD | `O(min(m, T)² · dh)` | `O(sketch · dh)` where sketch = `min(rank+oversample, q_cap)` |
| Decompression | Identical | Identical |
| Quality in free zone | Reference | Within `|Δ| ≤ 0.003` |
| Quality outside free zone | Reference | Slightly worse (capped sketch) |

FlashJoLT's `q_cap` policy:

```
q_cap = min(max(q_min(R), ⌈T / 32⌉), 512)
q_min(R) = 32 if R ≤ 4 else 64
```

For `T ≤ 1024`, the cap is `q_min(R)` — i.e. a no-op. FlashJoLT and
JoLT are bit-identical (modulo the sketch's random seed) for short
contexts. The cap grows as `O(log T)` for long contexts, which is
why FlashJoLT is the right choice for `T ≥ 8K`.

## Layer groups

The allocator can split the model into `layer_groups` independent
groups, each with its own `(r_T, r_d, b)` decision:

```python
enable_compression(model, method="flashjolt", target_memory="33%", layer_groups=4)
```

The paper's default is `layer_groups=1` and ablation A5 shows per-group
allocation beats uniform by up to **1 PPL at 4×**. In the free zone
(2-3×) the difference is within noise. **Use `layer_groups > 1` only
if you have a measured reason** (e.g. you have a heterogeneous model
with very different early vs late layer characteristics).

## Residual bit-widths

The allocator picks per-cell from the `bits` tuple. Default is
`(0, 2, 4, 8)` matching the paper.

```python
# Restrict to {0, 4} if your hardware doesn't support 2-bit ops efficiently.
enable_compression(model, method="flashjolt", target_memory="33%", bits=(0, 4))
```

The paper's bit-grid ablation (Table 3) shows quality saturates at
**4 bits**: 4 → 8 buys nothing measurable at 2×. 8-bit residual
also can't fit the budget at 2×, so 4 is the quality knee **and** the
only feasible default above 2×.

## Pre-RoPE vs post-RoPE key compression

JoLT compresses keys **before** RoPE and re-applies the rotation at
decode. The paper's Appendix C shows this is meaningfully better:

| Model | pre-RoPE | post-RoPE |
|---|---|---|
| Mistral-7B (2×) | 0.0095 rel err | 0.0162 rel err (+70%) |
| Mistral-7B (3×) | 0.0379 rel err | 0.0489 rel err (+29%) |
| LLaMA-2-13B (2×) | — | +22% to +44% |

Implementation note: the cache sees *unrotated* keys. The RoPE
application is downstream of the cache, so the model's standard
forward pass reapplies the rotation to the decompressed K. This is
the paper's recommended setting.

## Multi-GPU and device handling

The compressor runs on whatever device the cache tensors live on.
If you move the model to CUDA after `enable_compression`, the cache
follows it via `DynamicCache.update`'s standard device-tracking.

The HF adapter does not currently support tensor-parallel cache
sharding — that's an open problem for vLLM-mode deployment. Use the
vLLM adapter instead (see [research/paper_notes.md](research/paper_notes.md#vllm-integration)).

## Decode-time cost

Compression happens once at every cache update. Decompression
happens once at every cache read (i.e., every attention call). On
long contexts, decompression is the dominant cost; on short contexts
(T < 512), compression dominates.

The cache manager keeps the last appended slice in *uncompressed*
form when possible, so the first attention call after a write skips
the decompression step. This is most helpful for short contexts.

## Memory accounting

`handle.stats_dict()` returns:

```python
{
    "compress_calls": int,      # number of layer-write events
    "decompress_calls": int,    # number of layer-read events
    "bytes_original": int,      # uncompressed bytes across all events
    "bytes_compressed": int,    # bytes in the compressed payloads
    "compression_ratio": float, # bytes_original / bytes_compressed
    "memory_saved_bytes": int,  # bytes_original - bytes_compressed
}
```

These update live as `model.generate` runs. Use them in your eval
loop to record the actual compression ratio achieved.

## Common anti-patterns

1. **Don't change `model.dtype` after enabling.** The cache will
   re-cast but the compressor state was bound to the original dtype.

2. **Don't disable mid-generation.** The patched `DynamicCache` will
   route the in-flight reads to nowhere.

3. **Don't use `jolt` for `T ≥ 8K` contexts.** The exact SVD becomes
   the bottleneck. Use `flashjolt`.

4. **Don't expect `jolt` and `flashjolt` to be bit-identical at long
   contexts.** FlashJoLT's sketch is non-deterministic across seed
   choices. The cap policy is deterministic, but the sketch matrix
   within the cap is not.

5. **Don't fix `bits=(8,)` only.** Restricting the allocator to
   8-bit residual limits its flexibility at high ratios. Default
   `(0, 2, 4, 8)` is correct.

6. **Don't enable compression during training.** JoLT is inference-
   only.

## Profiling

```python
from kvcompress.runtime.profiler import CompressionProfiler

prof = CompressionProfiler()
with prof.record("compress", bytes_in=K.numel() * K.element_size()):
    kp, vp = comp.compress(K, V)

print(prof.summary())
# {'compress': {'count': 1, 'total_ms': 12.3, 'mean_ms': 12.3, ...}}
```

The profiler is passive (it doesn't change the algorithm) and can be
disabled via `prof._enabled = False` to remove the overhead entirely.

## What's next

- [user/long_context.md](user/long_context.md) — needle-in-haystack
  workflows
- [research/free_zone.md](research/free_zone.md) — when the free
  zone holds and when it doesn't
- [research/reproduction_notes.md](research/reproduction_notes.md) —
  what we did and didn't reproduce
- [benchmarks/overview.md](benchmarks/overview.md) — running
  benchmarks on your model