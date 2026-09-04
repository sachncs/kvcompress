# vLLM integration

kvfold ships with two vLLM integration paths. Both are optional and
require ``vllm>=0.7``.

## Shape A: ``export_kv`` / ``import_kv``

Save a model's KV cache to disk (safetensors), load it back. Useful
for warm-starting a vLLM worker from a precomputed cache, or moving
state between machines.

```python
from kvfold.adapter.vllm import export_kv, import_kv, is_vllm_available

if is_vllm_available():
    meta = export_kv(model, "/tmp/kv.safetensors", method="flash", ratio=3.0)
    # ... later ...
    import_kv(model, "/tmp/kv.safetensors", target_memory="50%")
```

* The exported file is a plain safetensors tensor store. No vLLM
  import is required to read the bytes — only the sidecar
  ``.meta.json``.
* The compressor method (``flash``, ``jolt``, ``int4``, ...) is
  recorded in the sidecar so the importer can rebuild.
* ``target_memory="100%"`` is a no-compression passthrough shortcut.

## Shape B: ``Offload`` vLLM OffloadHandler

Patch the vLLM V1 kv-offload worker so it compresses blocks as they
move between GPU and host memory.

```python
from kvfold.adapter import Offload
from kvfold.api import build_compressor

handler = Offload(build_compressor("flash", ratio=3.0))
# Register with vLLM via the vllm.v1.kv_offload worker hook
```

The handler implements the vLLM ``OffloadingHandler`` protocol; refer
to ``docs/dev/gpu.md`` for the manual verification steps (CUDA
required).

## Method names

Both helpers accept the same ``method`` keyword as
:func:`kvfold.api.enable_compression`:

| 0.2.0 method | Compression |
|--------------|--------------|
| ``jolt``     | Partial Tucker + JL-residual (paper-faithful) |
| ``flash``    | Randomised-SVD JoLT (paper-faithful) |
| ``low``      | Pure low-rank SVD baseline |
| ``int2`` / ``int4`` / ``int8`` | Per-cell integer quantisation |
| ``fp8``      | IEEE E4M3 / E5M2 quantisation |
| ``fp16`` / ``bf16`` | Dtype-only halving |
| ``pass``     | No compression (alias for the old ``identity``) |

## Migrating from 0.1.x

```python
# 0.1.x
export_kv(model, path, method="flashjolt", compression_ratio=3.0)

# 0.2.0
export_kv(model, path, method="flash", ratio=3.0)
```

See the :doc:`migration` page for the full table.
