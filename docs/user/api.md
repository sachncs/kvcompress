# API reference

The user-facing API is intentionally small: one entry point
([`enable_compression`](#enable_compression)) and one handle
([`CompressionHandle`](#compressionhandle)). Everything else lives in
`kvcompress.compressor` for users who want to build their own pipelines.

## `enable_compression`

```python
from kvcompress import enable_compression

handle = enable_compression(
    model,
    method="flashjolt",            # 'jolt', 'flashjolt', 'lowrank', 'int2', 'int4',
                                   # 'int8', 'fp8', 'fp16', 'bf16', 'identity'
    target_memory="25%",           # OR compression_ratio=4.0
    layer_groups=1,
    bits=(0, 2, 4, 8),
    seed=0,
)
```

### Args

| Name | Type | Description |
|---|---|---|
| `model` | `PreTrainedModel` | Hugging Face model to patch. |
| `method` | `str` | Compressor name (see table above). |
| `target_memory` | `str` or `float` | Target memory as a fraction (`0.25` or `"25%"`). Mutually exclusive with `compression_ratio`. |
| `compression_ratio` | `float` | Target ratio (e.g. `3.0` for 3×). Mutually exclusive with `target_memory`. |
| `layer_groups` | `int` | Number of contiguous layer groups for the allocator. Paper uses `1`; larger values give finer-grained control. |
| `bits` | `tuple[int]` | Allowed residual bit-widths. Default `(0, 2, 4, 8)`. |
| `cache_implementation` | `str` | HF cache implementation; we always use `dynamic` under the hood. |
| `seed` | `int` | Seed for randomized components. |

### Returns

`CompressionHandle` — see below.

### Raises

- `ValueError` if neither or both of `target_memory`/`compression_ratio` is provided.
- `NotImplementedError` if `method` is unknown.

## `CompressionHandle`

Returned by `enable_compression`.

```python
handle.stats_dict()  # {'compress_calls': 12, 'bytes_original': ..., ...}
handle.disable()     # restore original behaviour
```

### Methods

- `disable()` — reverts the patch. The model's behaviour returns to baseline.
- `stats_dict()` — returns a dict of cumulative statistics: compress calls, decompress calls, bytes original / compressed, ratio, memory saved.

### Attributes

- `adapter` — the underlying `HuggingFaceAdapter` instance (for power users).
- `model` — the patched model.
- `stats` — a mutable `CompressionStats` object updated by the patched DynamicCache.

## Core compression API

For users who want direct access to the compressor without the HF
adapter, every compressor implements the same ABC:

```python
from kvcompress import JoLTCompressor, FlashJoLTCompressor

comp = FlashJoLTCompressor(compression_ratio=3.0, bits=(0, 2, 4, 8))
key_payload, value_payload = comp.compress(key, value)
key_hat, value_hat = comp.decompress(key_payload, value_payload)
```

### `KVCompressor`

```python
class KVCompressor(ABC):
    name: str

    def compress(self, key, value) -> tuple[CompressedPayload, CompressedPayload]: ...
    def decompress(self, kp, vp) -> tuple[Tensor, Tensor]: ...
    def estimate_size(self, payload) -> int: ...
    def stats(self) -> dict[str, Any]: ...
```

### Concrete compressors

- `kvcompress.JoLTCompressor(compression_ratio=3.0, bits=(0, 2, 4, 8))`
- `kvcompress.FlashJoLTCompressor(compression_ratio=3.0, bits=(0, 2, 4, 8))`
- `kvcompress.IdentityCompressor()` — for ablation.
- `kvcompress.LowRankCompressor(rank=64)` — matrix SVD baseline.
- `kvcompress.IntQuantOnlyCompressor(bits=4)` — pure int quant.

## Cache API

```python
from kvcompress import CompressedKVCache, CacheManager, CompressionMetadata

cache = CompressedKVCache(compressor=comp)
cache.store(layer=0, key=k, value=v)
k_hat, v_hat = cache.retrieve(0)
cache.memory_used()      # bytes currently occupied
cache.compression_ratio()
cache.stats()
```

## Allocator API

```python
from kvcompress.compressor.allocator import JointAllocator, Cell

cells = [Cell(shape=(8, 256, 64), kind="key"), Cell(shape=(8, 256, 64), kind="value")]
alloc = JointAllocator(target_ratio=3.0)
result = alloc.optimize(cells)
result.allocations[0].r_token, result.allocations[0].r_feature, result.allocations[0].bits
```

## vLLM integration (Shape A)

```python
from kvcompress.adapters.vllm import export_kv, import_kv

# Save a vLLM / HF model's KV cache to disk in compressed form.
export_kv(model, "kv.safetensors", method="flashjolt", compression_ratio=3.0)

# Restore it (in this process or a different one).
import_kv(model, "kv.safetensors")
```

The exported file is a single safetensors with one tensor per (layer,
kind) cell plus a `.meta.json` sidecar. Both functions work with any HF
or vLLM-style model that exposes a `DynamicCache` via
`model.past_key_values` / `model.kv_cache` / `model.cache`.

For the deeper vLLM integration that hooks into the block-eviction
path, see `kvcompress.adapters.vllm_kv_offload.JoLTOffloadWorker`.

## CLI

```bash
kvcompress version
kvcompress validate [--skip-hf]
kvcompress benchmark [--suite all|memory|speed|reconstruction] [--output-dir PATH]
kvcompress profile --model MODEL_ID [--ratio 3.0]
kvcompress compress --model MODEL_ID --method flashjolt --target 33% --prompt "..."
```