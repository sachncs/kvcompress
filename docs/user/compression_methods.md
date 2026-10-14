# Compression methods

This page compares the compression methods `kvfold` ships and tells
you when to pick each one.

## Method summary

| Method | Speed | Quality | Compression ratio | When to use |
|---|---|---|---|---|
| `flash` | Fast | Tested quality varies by model | 2-4× | **Default.** Most use cases. |
| `jolt` | Slower | Paper-reported high fidelity | 2-4× | When you need exact reproduction. |
| `low` | Fast | Lossy | 2-8× | Ablation baseline; rare in production. |
| `int4` / `int8` | Fast | Lossy | 4-8× | When you want a fixed bit-width. |
| `fp8` | Fast | Lossless | 2× | Simple halving without algorithm change. |
| `fp16` / `bf16` | Fast | Lossless | 2× | Default of most models. |
| `pass` | Fastest | Identity | 1× | Ablation / debug. |

## JoLT vs FlashJoLT

The two are algorithmically identical except for the SVD path. JoLT uses
the *exact* token-mode SVD; FlashJoLT uses a *randomized* SVD capped at
``q_cap = min(max(q_min(R), ⌈T/32⌉), 512)``.

For ``T ≤ 1024``, FlashJoLT's cap policy is a no-op and the two give
bit-identical allocations. For longer contexts, FlashJoLT trades a tiny
amount of accuracy (≪ 0.003 relative Frobenius error in the free zone) for
a 5-13× compression-time speedup.

**Pick `flash` for production** unless you need exact reproducibility
or are running in the short-context regime where the difference is nil.

## Picking a ratio

The paper establishes a 2-3× *free zone* where JoLT is near-lossless on
both GQA (Mistral) and MHA (LLaMA) architectures. Outside the free zone,
GQA degrades gracefully but MHA falls off a cliff at 4-5×.

```python
# Recommended default for most workloads.
enable_compression(model, method="flash", target_memory=0.33)  # 3x

# Aggressive compression for batch / prompt-caching deployments.
enable_compression(model, method="flash", target_memory=0.20)  # 5x

# Maximum savings on long-context MHA models — expect quality loss.
enable_compression(model, method="flash", target_memory=0.12)  # 8x
```

See [Performance guide](performance_guide.md) for per-model guidance.
