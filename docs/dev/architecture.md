# Architecture

## Module map

```
                         ┌─────────────────┐
                         │  enable_compression│  (api.py)
                         └────────┬────────┘
                                  │ constructs
                                  ▼
                         ┌─────────────────┐
                         │ HuggingFaceAdapter│  (adapters/huggingface.py)
                         └────────┬────────┘
                                  │ owns
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
       ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
       │ KVCompressor │    │ CacheManager │    │ DynamicCache │
       │ (base.py)    │    │(cache/mgr.py)│    │  (patched)   │
       └──────┬──────┘    └──────┬───────┘    └──────────────┘
              │                   │
              │ uses              │ stores
              ▼                   ▼
       ┌─────────────┐    ┌──────────────────┐
       │   JoLT or   │    │ CompressedKVCache│
       │ FlashJoLT   │    │ (cache/compress) │
       └──────┬──────┘    └──────────────────┘
              │
              │ composes
              ▼
       ┌─────────────────┐
       │ Tucker + JL + Q │
       └─────────────────┘
```

## Data flow

A single `DynamicCache.update(key_states, value_states, layer_idx)`
call from the HF model generates one flow through the system:

1. **Parent update.** `super().update(...)` concatenates the new K/V onto
   the layer's existing tensors (HF's standard behaviour).
2. **Compress.** `KVCompressor.compress(key, value)` runs:
   - ST-HOSVD on each of K and V (token + feature modes truncated).
   - Residual = X - X_hat.
   - JL-rotate the residual.
   - Quantize the rotated residual at the allocator-chosen bit-width.
3. **Store.** `CacheManager.store(layer_idx, ...)` puts the resulting
   `CompressedPayload` objects into the layer-indexed cache.
4. **Decompress on read.** When the model calls
   `past_key_values[layer_idx]`, the patched `__getitem__` reconstructs
   K/V from the compressed payload and writes them back into the layer's
   tensors.

## Key abstractions

| Concept | Where | Why |
|---|---|---|
| `KVCompressor` | `compressor/base.py` | ABC for all compressors. New methods plug in here. |
| `CompressedPayload` | `compressor/base.py` | Self-describing payload with serializable metadata. |
| `CompressionMetadata` | `cache/metadata.py` | Layer-indexed summary for safetensors round-trips. |
| `Cell` | `compressor/allocator.py` | One (layer group, K/V) cell the allocator solves. |
| `Allocation` | `compressor/allocator.py` | One cell's (rT, rd, b) decision. |
| `HuggingFaceAdapter` | `adapters/huggingface.py` | Bridges HF cache ↔ KVCompressor. |
| Family shims | `adapters/<family>.py` | Per-family model patches (mostly no-ops today). |

## Algorithm data flow (one cell)

```
K, V ∈ R^{m × T × dh}
       │
       ▼  ST-HOSVD
(core_K, U_T, U_d) for K,  (core_V, U_T, U_d) for V
       │
       ▼  R = X - X̂
residual_K, residual_V ∈ R^{m × T × dh}
       │
       ▼  Π R^T (square rotation)
rotated_K, rotated_V ∈ R^{m·T × dh}
       │
       ▼  uniform b-bit quant
codes_K, codes_V ∈ R^{...} (packed uint8)
```

Decoding inverts the chain.

## Extension points

1. **New compressor**: subclass `KVCompressor`. Implement `compress` and
   `decompress`. Register in `api._build_compressor`. See
   [Adding a compressor](adding_a_compressor.md).
2. **New model family**: write `adapters/<name>.py` exposing `install`.
   Register in `adapters/registry.py`. See
   [Adding an adapter](adding_an_adapter.md).
3. **New quantization scheme**: subclass `IntQuantizer` (or implement
   the `Quantizer` protocol). Add a dispatch entry in
   `quantization.get_quantizer`.
4. **New cache backend**: subclass `CompressedKVCache`. The HF adapter
   only needs `store`, `retrieve`, and `memory_used`.
5. **New benchmark**: add a module to `benchmarks/` and wire it in
   `cli.py`'s `benchmark` command.