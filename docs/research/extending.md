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
# kvfold/core/tsvd.py
from kvfold.core.base import Compressor, Payload
import torch

class TSVDCompressor(Compressor):
    method = "t-svd"

    def compress(self, key, value):
        # ... t-SVD implementation ...
        return kp, vp

    def restore(self, kp, vp):
        # ... inverse t-SVD ...
        return k, v
```

Then register in `kvfold/core/builtins.py`:

```python
from kvfold.core.dispatch import REGISTRY
from kvfold.core.tsvd import TSVDCompressor

REGISTRY.register("t-svd", TSVDCompressor)
```

## Custom quantizers

Subclass `IntQuantImpl` or implement the `Quantizer` protocol:

```python
from kvfold.core.quant import IntQuant

class Int4PerGroup(IntQuant):
    """INT4 with per-group scales of 32."""

    def __init__(self, group_size: int = 32):
        super().__init__(bits=4, symmetric=True, per_channel=False, group_size=group_size)
```

Register in `kvfold.core.quant.get_quantizer`:

```python
def get_quantizer(name, **kwargs):
    if name == "int4-group-32":
        return Int4PerGroup(group_size=kwargs.get("group_size", 32))
    ...
```

## Hybrid methods

Combine JoLT with another compressor by composing in `compress`:

```python
from kvfold.core.base import Compressor, Payload
from kvfold import Jolt, IntQuant

class JoLTThenInt4(Compressor):
    """Run JoLT first, then quantize the result's residual to INT4."""

    method = "jolt+int4"

    def __init__(self, **kwargs):
        super().__init__()
        self.jolt = Jolt(**kwargs)
        self.int4 = IntQuant(bits=4)

    def compress(self, key, value):
        # Apply JoLT first; the residual becomes the data to quantize.
        k_jolt, v_jolt = self.jolt.compress(key, value)
        # ... extract residual factors, quantize, wrap ...
        ...

    def restore(self, kp, vp):
        ...
```

Register via `kvfold/core/builtins.py`:

```python
from kvfold.core.dispatch import REGISTRY
from somewhere import JoLTThenInt4

REGISTRY.register("jolt+int4", JoLTThenInt4)
```

## Adding a new family

See [Adding an adapter](../dev/adding_an_adapter.md).

## Adding a new cache backend

Subclass `Cache` and override `store`, `retrieve`, and
`memory_used`. The HF adapter only needs these three methods plus
`__contains__` and `__len__`.

```python
from kvfold.store.compress import Cache
import torch

class DiskBackedCache(Cache):
    def store(self, layer, key, value, **kwargs):
        payload = self.compressor.compress(key, value)
        path = self._dir / f"layer_{layer}.pt"
        torch.save(payload, path)
        ...
```

The HF adapter will pick up your cache automatically if you pass it via
`HF(..., cache=my_cache)`.
