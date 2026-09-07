# Manual GPU verification

The kvfold test suite runs entirely on CPU; GPU-dependent paths are
verified manually on a CUDA machine. This document records the
commands to run when validating a new release on hardware.

## Triton kernel path

The Triton-fused Tucker reconstruction kernel lives in
``kvfold/kernel/triton/tucker.py``. It compiles lazily on first use
and caches the compiled artifact on the active CUDA device.

To verify on a CUDA machine:

```bash
# 1. Install the optional GPU extra
pip install "kvfold[triton]"

# 2. Run the dedicated parity test
pytest -v tests/unit/test_kernel_parity.py -m gpu
```

Expected output: ``backend='triton'`` matches ``backend='torch'`` to
within ``1e-5`` relative Frobenius error on a synthetic
``(m=8, T=1024, dh=128)`` tensor. Skip on CPU.

## vLLM Offload handler (Shape B)

```bash
pip install "kvfold[vllm]"
pytest -v tests/integration/test_vllm_offload.py
```

The handler installs against ``vllm.v1.kv_offload.worker.OffloadingHandler``
and runs an end-to-end block compress/restore cycle with the
vLLM offload worker.

## Compression profiler

```bash
python -m kvfold.bench.speed --m 8 --T 1024 --dh 128 --ratio 3
```

Records compress/restore ms per method. Triton backend should be
5–13× faster than the exact JoLT path on long contexts
(T ≥ 2048); CPU baseline.

## Reporting

If any of the above fails on hardware, open an issue with the
``gpu-verification`` label and attach the test output.
