# Debugging

## Enable verbose logging

```python
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

# kvcompress-specific debug
logging.getLogger("kvcompress").setLevel(logging.DEBUG)
```

You should see allocator decisions (`lambda_star`, `achieved_ratio`),
compression timings, and family shim dispatch.

## Inspect a CompressedKVCache

```python
from kvcompress import CompressedKVCache

cache = ...  # wherever you got it
cache.stats()                        # bytes_original, bytes_compressed, n_layers
cache.metadata()                     # CompressionMetadata
m = cache.metadata()
for entry in m.layers:
    print(entry.layer, entry.kind, entry.r_token, entry.r_feature, entry.bits)
```

## Inspect a compressor's stats

```python
comp = JoLTCompressor(compression_ratio=3.0)
kp, vp = comp.compress(K, V)
print(comp.stats())
# {'method': 'jolt', 'call_count': 1, 'compress_time_ms': 23.4,
#  'bytes_original': ..., 'bytes_compressed': ..., ...}
```

## Profile a session

```python
from kvcompress.runtime.profiler import CompressionProfiler

prof = CompressionProfiler()
with prof.record("compress_one", bytes_in=K.numel() * K.element_size()):
    kp, vp = comp.compress(K, V)
print(prof.summary())
```

## Verify the DynamicCache patch is active

```python
import transformers.cache_utils as cu
print("DynamicCache class:", cu.DynamicCache)
# Should print <class 'kvcompress.adapters.huggingface.HuggingFaceAdapter._install_dynamic_cache.<locals>._KvCompressCache'>
# after enable_compression.
```

If it prints the original `DynamicCache`, the patch wasn't installed
(probably because the model was loaded before `enable_compression` was
called, or a transformers-internal import happened before the patch).
Re-import transformers and try again.

## Common pitfalls

- **Reading the patched class via `from … import DynamicCache`.** Use
  `transformers.cache_utils.DynamicCache` to see the live symbol.
- **Calling `model.eval()` after `enable_compression`.** The patch
  survives mode changes, but if you reassign the model (e.g. via
  `torch.compile`) the cache may be re-created.
- **Disabling in the middle of generation.** The patched DynamicCache
  will be replaced and in-flight reads will fail. Always disable between
  generations.

## Reporting a bug

Please open an issue with:

- `kvcompress --version` output.
- Python, PyTorch, transformers versions (`python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"`).
- Hardware (CPU / GPU / MPS).
- Minimal reproduction script.