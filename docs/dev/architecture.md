# Architecture

## Module map

```
                         ┌──────────────────┐
                         │ enable_compression│  (api.py)
                         └────────┬─────────┘
                                  │ constructs
                                  ▼
                         ┌──────────────────┐
                         │ HF               │  (adapter/huggingface.py)
                         └────────┬─────────┘
                                  │ owns
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
       ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
       │ Compressor  │    │ Pool         │    │ DynamicCache │
       │ (core/base) │    │(store/manager│    │  (patched)   │
       └──────┬──────┘    └──────┬───────┘    └─────────────┘
              │                  │
              │ uses             │ stores
              ▼                  ▼
       ┌─────────────┐    ┌──────────────────┐
       │ Jolt / Flash │    │ Cache            │
       │             │    │ (store/compress) │
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
2. **Compress.** `Compressor.compress(key, value)` runs:
   - ST-HOSVD on each of K and V (token + feature modes truncated).
   - Residual = X - X_hat.
   - JL-rotate the residual.
   - Quantize the rotated residual at the allocator-chosen bit-width.
3. **Store.** `Pool.store(layer_idx, ...)` puts the resulting `Payload`
   objects into the layer-indexed cache.
4. **Decompress on read.** When the model calls
   `past_key_values[layer_idx]`, the patched `__getitem__` reconstructs
   K/V from the compressed payload and writes them back into the layer's
   tensors.

## Key abstractions

| Concept | Where | Why |
|---|---|---|
| `Compressor` | `kvfold/core/base.py` | ABC for all compressors. New methods plug in here. |
| `Payload` | `kvfold/core/base.py` | Self-describing payload with serializable metadata. |
| `Meta` | `kvfold/store/metadata.py` | Layer-indexed summary for safetensors round-trips. |
| `Cell` | `kvfold/core/budget.py` | One (layer group, K/V) cell the allocator solves. |
| `Pick` | `kvfold/core/budget.py` | One cell's (rT, rd, b) decision. |
| `HF` | `kvfold/adapter/huggingface.py` | Bridges HF cache ↔ Compressor. |
| Family shims | `kvfold/adapter/registry.py` | Per-family model patches (mostly no-ops today). |

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

1. **New compressor**: subclass `Compressor`. Implement `compress` and
   `restore`. Register via `from kvfold.core.dispatch import register as
   _register; @_register("mymethod", MyClass)`. See
   [Adding a compressor](adding_a_compressor.md).
2. **New model family**: write `adapter/<name>.py` exposing `install`.
   Register in `adapter/registry.py`. See
   [Adding an adapter](adding_an_adapter.md).
3. **New quantization scheme**: subclass `IntQuantImpl` (or implement
   the `Quantizer` protocol). Add a dispatch entry in
   `core/quantization.get_quantizer`.
4. **New cache backend**: subclass `Cache`. The HF adapter only needs
   `store`, `retrieve`, and `memory_used`.
5. **New benchmark**: add a module to `bench/` and wire it in
   `cli.py`'s `benchmark` command.
