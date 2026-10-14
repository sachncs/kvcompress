# Reproducibility contract

For a fixed input tensor, method configuration, seed, device, PyTorch version,
and thread count, kvfold's randomized compressors are expected to produce
equivalent payload metadata and numerically equivalent reconstructions within
the method's tolerance. This is a same-environment contract; byte-identical
payloads across PyTorch, CUDA, or GPU architectures are not promised.

To record an experiment:

```python
import torch

torch.manual_seed(0)
torch.set_num_threads(1)
compressor = build_compressor("flash", ratio=3.0, seed=0)
```

Also record Python, PyTorch, CUDA, device, dtype, tensor shape, method,
configuration, seed, and thread count. Benchmark scripts emit this metadata
automatically in their JSON artifacts.

For strict CUDA experiments, enable the relevant PyTorch deterministic
settings and expect a performance trade-off. The repository's CPU tests do
not certify CUDA determinism or Triton parity; those require the GPU release
lane described in `gpu.md`.
