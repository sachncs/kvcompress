# Migration guide: 0.1.x → 0.2.0

0.2.0 is a clean break from 0.1.x. The five stable public names
survive; everything else was renamed.

## Public API surface

The following names keep their spelling but their signatures changed:

| 0.1.x call | 0.2.0 call |
|------------|------------|
| `enable_compression(model, method="flashjolt", compression_ratio=3.0)` | `enable_compression(model, method="flash", ratio=3.0)` |
| `enable_compression(model, method="jolt", compression_ratio=3.0)` | `enable_compression(model, method="jolt", ratio=3.0)` |
| `enable_compression(model, method="lowrank", rank=64)` | `enable_compression(model, method="low", rank=64)` |
| `enable_compression(model, method="identity")` | `enable_compression(model, method="pass")` |
| `enable_compression(model, method="fp8")` | now real IEEE E4M3/E5M2 quantisation |
| `build_compressor(method, compression_ratio=3.0)` | `build_compressor(method, ratio=3.0)` |
| `handle.stats_dict()["compression_ratio"]` | `handle.stats_dict()["ratio"]` |

Unchanged:

- `enable_compression` / `disable_compression`
- `CompressionHandle` / `parse_target_memory`
- `build_compressor` / `supported_methods` (function names)
- `MethodName` (still exported from `kvfold.api`)

## Internal symbols

Every internal class and function was renamed to a single word.
There are **no backward shims**. Use this table to plan your migration:

### Compressors

| 0.1.x | 0.2.0 |
|-------|-------|
| `kvfold.compressor.KVCompressor` | `kvfold.core.Compressor` |
| `kvfold.compressor.CompressedPayload` | `kvfold.core.Payload` |
| `kvfold.compressor.CompressorStats` | `kvfold.core.Stats` |
| `kvfold.compressor.JoLTCompressor` | `kvfold.core.Jolt` |
| `kvfold.compressor.FlashJoLTCompressor` | `kvfold.core.Flash` |
| `kvfold.compressor.LowRankCompressor` | `kvfold.core.Low` |
| `kvfold.compressor.IdentityCompressor` | `kvfold.core.Pass` |
| `kvfold.compressor.IntQuantOnlyCompressor` | `kvfold.core.IntQuant` |
| `kvfold.compressor.quantization_only.IntQuantOnlyCompressor` | `kvfold.core.IntQuant` |
| `kvfold.compressor.quantization.IntQuantizer` | `kvfold.core.IntQuant` (quantizer) |
| `kvfold.compressor.quantization.FloatCastQuantizer` | `kvfold.core.FloatCast` (quantizer) |
| `kvfold.compressor.identity.IdentityCompressor` | `kvfold.core.Pass` |

### Adapters

| 0.1.x | 0.2.0 |
|-------|-------|
| `kvfold.adapters.HuggingFaceAdapter` | `kvfold.adapter.HF` |
| `kvfold.adapters.vllm_kv_offload.JoLTOffloadHandler` | `kvfold.adapter.Offload` |
| `kvfold.adapters.vllm_kv_offload.ThreadSafeEvictionPool` | `kvfold.adapter.EvictPool` |

### Storage

| 0.1.x | 0.2.0 |
|-------|-------|
| `kvfold.cache.CompressedKVCache` | `kvfold.store.Cache` |
| `kvfold.cache.CacheManager` | `kvfold.store.Pool` |
| `kvfold.cache.CompressionMetadata` | `kvfold.store.Meta` |
| `kvfold.cache.LayerCompression` | `kvfold.store.LayerMeta` |

### Allocator

| 0.1.x | 0.2.0 |
|-------|-------|
| `kvfold.compressor.JointAllocator` | `kvfold.core.Bisect` |
| `kvfold.compressor.GreedyAllocator` | `kvfold.core.Greedy` |
| `kvfold.compressor.Allocation` | `kvfold.core.Pick` |
| `kvfold.compressor.AllocationResult` | `kvfold.core.Plan` |

### Math primitives

| 0.1.x | 0.2.0 |
|-------|-------|
| `kvfold.compressor.SVD` (class) | `kvfold.core.Decomposer` (ABC) + `Exact` / `Randomized` |
| `kvfold.compressor.SVDResult` | `kvfold.core.Decomposition` |
| `kvfold.compressor.JLProjection` | `kvfold.core.Projector` (ABC) + `Gaussian` / `Rademacher` / `Sparse` |
| `kvfold.compressor.JLDistribution` | `kvfold.core.Distribution` (type alias) |
| `kvfold.compressor.TuckerFactors` | `kvfold.core.Tucker` |
| `kvfold.compressor.ResidualPayload` | `kvfold.core.Residual` |

### Runtime

| 0.1.x | 0.2.0 |
|-------|-------|
| `kvfold.runtime.MemoryPool` | `kvfold.runtime.Pool` |
| `kvfold.runtime.CompressionProfiler` | `kvfold.runtime.Profile` |

## Method names

| 0.1.x method | 0.2.0 method |
|--------------|--------------|
| `flashjolt` | `flash` |
| `lowrank` | `low` |
| `identity` | `pass` |

The other method names (`jolt`, `int2`, `int4`, `int8`, `fp8`, `fp16`)
are unchanged. `bf16` is newly supported.

## Auto-fix

```bash
# dry-run
sed -i.bak \
  -e 's/HuggingFaceAdapter/HF/g' \
  -e 's/JointAllocator/Bisect/g' \
  -e 's/GreedyAllocator/Greedy/g' \
  -e 's/CompressionMetadata/Meta/g' \
  -e 's/CacheManager/Pool/g' \
  -e 's/CompressedKVCache/Cache/g' \
  -e 's/compression_ratio/ratio/g' \
  -e 's/\.decompress(/\.restore(/g' \
  your_code.py

# review the diff, then delete the backup
```
