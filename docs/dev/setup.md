# Setup

## Clone and install

```bash
git clone https://github.com/sachncs/kvcompress
cd kvcompress
pip install -e ".[dev,bench,docs]"
```

This installs:

- The package in editable mode.
- Test dependencies (`pytest`, `pytest-cov`, `hypothesis`).
- Lint/format (`ruff`) and type check (`mypy`).
- Documentation (`mkdocs`).
- Benchmark extras (`matplotlib`, `pandas`, `datasets`).

## Running tests

```bash
# Fast unit tests only (no model downloads).
pytest -m "not slow and not integration and not gpu"

# All tests including integration.
pytest

# Coverage.
pytest --cov=kvcompress --cov-report=term-missing
```

The first run downloads a 500 MB GPT-2 model for the integration test
fixtures. Subsequent runs use the cached copy.

## Lint, format, type-check

```bash
ruff check src tests examples scripts
ruff format src tests examples scripts
mypy src
```

These three are also run by GitHub Actions on every push.

## Project layout

```
kvcompress/
├── api.py                    # enable_compression + CompressionHandle
├── cli.py                    # Typer app
├── compressor/
│   ├── base.py               # KVCompressor ABC
│   ├── jolt.py               # JoLTCompressor
│   ├── flashjolt.py          # FlashJoLTCompressor
│   ├── lowrank.py            # LowRankCompressor (baseline)
│   ├── quantization_only.py  # IntQuantOnlyCompressor (baseline)
│   ├── identity.py           # IdentityCompressor
│   ├── tucker.py             # partial ST-HOSVD
│   ├── svd.py                # SVD class (exact + randomized)
│   ├── jl.py                 # Johnson-Lindenstrauss projections
│   ├── quantization.py       # FP16/BF16/FP8/INT2/4/8 quantizers
│   ├── residual.py           # encode/decode residual
│   └── allocator.py          # JointAllocator + GreedyAllocator
├── cache/
│   ├── compress.py           # CompressedKVCache
│   ├── manager.py            # CacheManager
│   └── metadata.py           # CompressionMetadata
├── adapters/
│   ├── huggingface.py        # HuggingFaceAdapter
│   ├── registry.py           # model_type -> shim
│   ├── vllm.py               # vLLM adapter
│   ├── llama.py              # family shims
│   ├── mistral.py
│   ├── qwen.py
│   ├── gemma.py
│   ├── phi.py
│   ├── mixtral.py
│   ├── falcon.py
│   ├── deepseek.py
│   └── internlm.py
├── runtime/
│   ├── memory.py             # MemoryPool
│   └── profiler.py           # CompressionProfiler
├── kernels/
│   ├── triton/compression.py # Fused kernels (PyTorch fallback)
│   └── triton/tucker_reconstruct.py  # Real Triton kernel
└── benchmarks/
    ├── memory.py
    ├── throughput.py
    ├── reconstruction.py
    └── plot.py

tests/
├── unit/
├── integration/
├── property/
└── fixtures/

scripts/
├── run_table2_reconstruction.py
├── run_memory_benchmark.py
├── run_speed_benchmark.py
├── validate_install.py
├── run_needle_haystack.py
├── sweep_compression_ratio.py
├── profile_model.py
└── reproduce_paper_numbers.sh

examples/                        # Jupyter-friendly demos
docs/
├── user/
├── dev/
├── research/
└── benchmarks/
```