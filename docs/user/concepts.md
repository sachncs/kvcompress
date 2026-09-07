# Concepts

This page explains the building blocks the rest of the documentation
assumes you know. If you're new to KV-cache compression, read this top to
bottom; if you've worked with Palu, KIVI, or xKV before, skim the
*allocator* section.

## KV cache

Every autoregressive transformer caches the key and value projections of
all previous tokens so it doesn't have to recompute them. For a model with
``L`` layers, ``B`` batch size, context length ``T``, and per-head dim
``dh``, the cache is ``O(B · T · L · n_kv · dh)`` floats.

Once ``T`` is large, the cache becomes the dominant memory cost — more
than the model weights themselves. The cache is also persistent: prompt
caching, beam search, and parallel sampling all keep many cached prefixes
alive at once.

## Tucker decomposition

A 3-D tensor ``X ∈ R^{m × T × d}`` can be approximated by Tucker
decomposition as a small *core* tensor ``G ∈ R^{m × r_T × r_d}`` multiplied
by three basis matrices ``U^{(m)}, U^{(T)}, U^{(d)}``. JoLT uses
*partial* Tucker: identity bases on the head/layer axis (mode 0), with
rank truncation only on the token and feature modes. ST-HOSVD computes
this in one pass by truncating each mode's leading singular subspace.

## Johnson-Lindenstrauss projection

A random Gaussian or ±1 matrix ``Π ∈ R^{d × d}`` is approximately
distance-preserving: for any vector ``x``, ``||Πx|| ≈ ||x||``. JoLT uses
JL to *rotate* the residual ``R = X - X̂`` before quantizing it; the
rotated residual has near-uniform energy distribution, so low-bit
quantization is uniform in quality across all dimensions.

## Joint allocator

For a target byte budget ``B``, the allocator chooses per-(layer group,
K/V) (r_token, r_feature, residual_bits) that minimize reconstruction
error. The Lagrangian relaxation ``L(λ) = Σ [e + λ·s]`` decouples across
cells, so each cell is solved by an exhaustive grid search and ``λ`` is
found by bisection to hit the global budget.

The cost and error model follow the paper (Eqs. 1 and 2):

```
s_{g,t}(rT, rd, b) = (m·rT·rd + T·rT + dh·rd) · 2 + (b/8) · m · T · dh   (bytes)
e_{g,t}(rT, rd, b) ≈ ε²(b) · τ_{g,t}(rT, rd)                          (error)
```

## Compression methods

`kvcompress` ships several methods, all of which implement the same
`KVCompressor` ABC:

| Method | Description |
|---|---|
| `jolt` | Paper-faithful JoLT (exact token-mode SVD). |
| `flashjolt` | Randomized-SVD variant. **Default.** |
| `lowrank` | Matrix SVD baseline (2-D factorization). |
| `int2`/`int4`/`int8` | Pure integer quantization. |
| `fp8` | FP8 storage. |
| `fp16`/`bf16` | Half-precision storage. |
| `identity` | No compression (passthrough). |

## Cap-implementation detail

The Hugging Face adapter works by patching `DynamicCache`. Every
`DynamicCache.update` call (which happens once per layer per generation
step) compresses the just-appended K/V slice. The next attention call
triggers `DynamicCache.__getitem__`, which decompresses back into the
layer's K/V tensors. The model then runs its standard attention code.

Compression happens on the **CPU/GPU** the cache is on. If you move the
model to CUDA after `enable_compression`, you may also need to move the
cache — see [Performance guide](performance_guide.md).