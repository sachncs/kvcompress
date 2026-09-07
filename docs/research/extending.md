# Extending

How to extend JoLT with new multilinear formats, custom quantizers, or
hybrid methods.

## Alternative multilinear formats

The paper's Appendix B.2 compares four formats on Mistral and LLaMA:

| Ratio | Tucker | CP | t-SVD | TT |
|---|---|---|---|---|
| 2× (Mistral K) | 0.087 | 0.098 | 0.147 | 0.224 |
| 2× (Mistral V) | 0.279 | 0.301 | 0.427 | 0.552 |
| 3× (Mistral K) | 0.125 | 0.155 | 0.204 | 0.303 |

Tucker wins at every ratio; CP is intractable on the larger MHA tensor;
t-SVD and TT are dominated.

If you want to plug in another format:

```python
# kvcompress/compressor/tsvd.py
from kvcompress.compressor.base import KVCompressor, CompressedPayload
import torch

class TSVDCompressor(KVCompressor):
    name = "t-svd"

    def compress(self, key, value):
        # ... t-SVD implementation ...
        return kp, vp

    def decompress(self, kp, vp):
        # ... inverse t-SVD ...
        return k, v
```

Then register in `api._build_compressor`:

```python
def _build_compressor(method, **kwargs):
    method = method.lower()
    if method == "t-svd":
        from kvcompress.compressor.tsvd import TSVDCompressor
        return TSVDCompressor(**kwargs)
    ...
```

## Custom quantizers

Subclass `IntQuantizer` or implement the `Quantizer` protocol:

```python
from kvcompress.compressor.quantization import IntQuantizer

class Int4PerGroup(IntQuantizer):
    """INT4 with per-group scales of 32."""

    def __init__(self, group_size=32):
        super().__init__(bits=4, symmetric=True, per_channel=False, group_size=group_size)
```

Register in `quantization.get_quantizer`:

```python
def get_quantizer(name, **kwargs):
    if name == "int4-group-32":
        return Int4PerGroup(group_size=kwargs.get("group_size", 32))
    ...
```

## Hybrid methods

Combine JoLT with another compressor by composing in `compress`:

```python
class JoLTThenInt4(KVCompressor):
    """Run JoLT first, then quantize the result's residual to INT4."""

    name = "jolt+int4"

    def __init__(self, **kwargs):
        super().__init__()
        self.jolt = JoLTCompressor(**kwargs)
        self.int4 = IntQuantOnlyCompressor(bits=4)

    def compress(self, key, value):
        # Apply JoLT first; the residual becomes the data to quantize.
        k_jolt, v_jolt = self.jolt.compress(key, value)
        # ... extract residual factors, quantize, wrap ...
        ...

    def decompress(self, kp, vp):
        ...
```

## Adding a new family

See [Adding an adapter](../dev/adding_an_adapter.md).

## Adding a new cache backend

Subclass `CompressedKVCache` and override `store`, `retrieve`, and
`memory_used`. The HF adapter only needs these three methods plus
`__contains__` and `__len__`.

```python
class DiskCompressedKVCache(CompressedKVCache):
    def store(self, layer, key, value, **kwargs):
        payload = self._compressor.compress(key, value)
        path = self._dir / f"layer_{layer}.pt"
        torch.save(payload, path)
        ...
```

The HF adapter will pick up your cache automatically if you pass it via
`HuggingFaceAdapter(..., cache=my_cache)`.