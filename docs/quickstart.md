# Quickstart

The shortest path from `pip install` to a compressed model.

## Install

```bash
pip install kvcompress
```

Optional extras (recommended for the full experience):

```bash
pip install "kvcompress[bench]"      # matplotlib, datasets, pandas
pip install "kvcompress[dev]"        # pytest, ruff, mypy, hypothesis
pip install "kvcompress[triton]"     # Triton kernels (NVIDIA only)
pip install "kvcompress[vllm]"       # vLLM integration
```

## The 30-second example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvcompress import enable_compression

# Load any causal LM.
model = AutoModelForCausalLM.from_pretrained("gpt2")
handle = enable_compression(model, method="flashjolt", target_memory="33%")

# Generate as usual. The KV cache is compressed transparently.
tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok.encode("The capital of France is", return_tensors="pt")
out = model.generate(ids, max_new_tokens=20)
print(tok.decode(out[0]))
print("compression stats:", handle.stats_dict())

handle.disable()  # restore original behaviour
```

The `target_memory="33%"` argument is equivalent to `compression_ratio=3.0`
(33% of original = 1/3 = 3× compression). You can use either.

## What `enable_compression` actually does

The function returns a `CompressionHandle`. While the handle is active:

1. Every `DynamicCache.update(K, V, layer_idx)` call is intercepted.
   The model writes its K and V to the cache as usual; the patched
   `DynamicCache.__init__` subclass compresses them and stores the
   compressed payload in our `CacheManager`.
2. Every `cache[layer_idx]` (i.e. the attention read) is intercepted.
   The patched `__getitem__` decompresses the payload and writes the
   reconstructed K/V back to the layer's tensors.

The model code itself doesn't change.

## Picking a method

| `method=` | Speed | Quality | Compression |
|---|---|---|---|
| `"flashjolt"` | Fast | Near-lossless | 2-4× |
| `"jolt"` | Slower (exact SVD) | Near-lossless | 2-4× |
| `"lowrank"` | Fast | Lossy | 2-8× |
| `"int4"` / `"int8"` | Fast | Lossy | 4-8× |
| `"fp8"` | Fast | Lossless (just cast) | 2× |
| `"identity"` | Fastest | Identity (fp16 cast) | 2× (just dtype) |

**Default: `flashjolt`.** It's the paper's recommended fast variant.
Use `jolt` only if you need exact reproducibility; in practice
`flashjolt` matches `jolt` within `|Δ| ≤ 0.003` in the free zone.

## Picking a ratio

The paper's "free zone" is **2-3×** for both GQA and MHA models. Within
this band, the model's perplexity drifts by < 1% (statistical noise).
Outside:

- **GQA models** (Mistral, Qwen, Gemma2) degrade gracefully. 4-8× is
  still acceptable for many workloads.
- **MHA models** (LLaMA, Phi) fall off a cliff at 4-5×. Stay in 2-3×.

```python
# Default: 3× (free zone)
enable_compression(model, method="flashjolt", target_memory="33%")

# Aggressive for GQA
enable_compression(model, method="flashjolt", target_memory="20%")  # 5×

# Maximum savings (lossy)
enable_compression(model, method="flashjolt", target_memory="12%")  # 8×
```

You can also use absolute bytes:

```python
enable_compression(model, method="flashjolt", compression_ratio=3.0)
```

## Inspecting what happened

```python
handle = enable_compression(model, method="flashjolt", target_memory="33%")
model.generate(...)  # run as usual

# Cumulative stats
print(handle.stats_dict())
# {'compress_calls': 12, 'decompress_calls': 12,
#  'bytes_original': 3145728, 'bytes_compressed': 524288,
#  'compression_ratio': 6.0, 'memory_saved_bytes': 2621440}
```

`compress_calls` is the number of layer-write events. `decompress_calls`
is the number of layer-read events. They should track the number of
forward passes in `generate`.

## Direct compression (no HF model)

For tests, ablations, or building your own integration, the algorithm
is usable directly:

```python
import torch
from kvcompress import JoLTCompressor

# A KV cache slice: (merged_heads, tokens, head_dim)
K = torch.randn(8, 256, 64)
V = torch.randn(8, 256, 64)

comp = JoLTCompressor(compression_ratio=3.0, bits=(0, 2, 4, 8))
k_payload, v_payload = comp.compress(K, V)
K_hat, V_hat = comp.decompress(k_payload, v_payload)

# Reconstruction error (small at 3× on smooth spectra)
rel_err = (K - K_hat).norm() / K.norm()
print(f"K rel err: {rel_err.item():.3f}")
```

The compressed payloads are JSON-serialisable; see
[user/api.md](user/api.md#core-compression-api) for the full surface.

## What's next

- [user/api.md](user/api.md) — every public symbol
- [user/performance_guide.md](user/performance_guide.md) — picking
  ratios for production
- [user/long_context.md](user/long_context.md) — needle-in-haystack
  workflows
- [examples/02_direct_compression.py](../examples/02_direct_compression.py)
  — full ablation example

## Common pitfalls

1. **Don't enable during training.** JoLT is inference-only; the
   cache writer doesn't track gradients.
2. **Don't change `model.dtype` after enabling.** The compressor
   uses the cache's dtype at enable time.
3. **Don't disable mid-generation.** The patched `DynamicCache` will
   route the in-flight reads to nowhere.
4. **For long contexts (≥ 8K), use `flashjolt` not `jolt`.** FlashJoLT's
   randomised SVD with the cap policy scales O(T) → O(T log T).

See [user/troubleshooting.md](user/troubleshooting.md) for error-message
decoding.