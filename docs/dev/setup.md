# Setup

## Clone and install

```bash
git clone https://github.com/sachncs/kvcompress
cd kvcompress
uv sync --locked --extra dev --extra bench --extra docs
```

This installs into the project-managed `.venv`:

- The package in editable mode.
- Test dependencies (`pytest`, `pytest-cov`, `hypothesis`).
- Lint/format (`ruff`) and type check (`mypy`).
- Google-style docstring and convention checks (`pydocstyle`).
- Documentation (`mkdocs`).
- Benchmark extras (`matplotlib`, `pandas`, `datasets`).

## Running tests

```bash
# Fast unit tests only (no model downloads).
uv run pytest -m "not slow and not integration and not gpu"

# All tests including integration.
uv run pytest

# Coverage.
uv run pytest --cov=kvfold --cov-report=term-missing
```

The first run downloads a 500 MB GPT-2 model for the integration test
fixtures. Subsequent runs use the cached copy.

## Lint, format, type-check

```bash
uv run ruff check kvfold tests examples scripts
uv run ruff format --check kvfold tests examples scripts
uv run mypy kvfold
uv run pydocstyle --convention=google --match='.*\\.py' --add-ignore=D100,D104,D105,D107 kvfold
uv run python scripts/check_conventions.py
```

These three are also run by GitHub Actions on every push.

## Project layout

```
kvfold/
├── __init__.py               # Lazy-export public surface (LAZY_EXPORTS)
├── api.py                    # enable_compression + CompressionHandle
├── cli.py                    # Typer app
├── config.py                 # Typed config objects + MethodConfigRegistry
├── errors.py                 # Error hierarchy
├── adapter/
│   ├── base.py
│   ├── huggingface.py        # HF adapter + DynamicCache interception
│   ├── registry.py           # model_type -> Family
│   ├── vllm.py               # vLLM Shape A: export_kv / import_kv
│   └── vllm_offload.py       # VLLM Shape B: Offload
├── core/
│   ├── base.py               # Compressor ABC + Payload + Stats
│   ├── identity.py           # Pass (passthrough baseline)
│   ├── low.py                # Low (matrix SVD baseline)
│   ├── int_quant.py          # IntQuant (per-channel int quantisation)
│   ├── float_cast.py         # FloatCast (dtype-halving)
│   ├── float8.py             # Float8 (IEEE E4M3/E5M2)
│   ├── quant.py              # int/fp quantiser primitives + bit-packing
│   ├── residual.py           # encode/decode residual
│   ├── tucker.py             # partial ST-HOSVD + reconstruction
│   ├── svd.py                # Decomposer (Exact + Randomized)
│   ├── jl.py                 # Johnson-Lindenstrauss projections + cache
│   ├── budget.py             # Bisect + Greedy allocators
│   ├── dispatch.py           # Compressor dispatcher + registry
│   ├── jolt.py               # Jolt
│   ├── flash.py              # Flash (randomised mode-1)
│   └── builtins.py           # wires concrete compressors into the dispatcher
├── store/
│   ├── compress.py           # Cache
│   ├── manager.py            # Pool facade
│   └── metadata.py           # Meta + LayerMeta
├── runtime/
│   ├── pool.py               # MemoryPool
│   └── profile.py            # CompressionProfiler
├── kernel/
│   └── triton/
│       └── tucker.py         # Fused Triton kernel (PyTorch fallback)
└── bench/
    ├── memory.py             # Bytes-per-method sweep
    ├── table2.py             # Paper Table 2 reproduction
    ├── speed.py              # Compress / decompress wall-time
    └── plot.py               # Matplotlib bar charts

tests/
├── unit/
├── integration/
├── property/
├── regression/
└── fixtures/

scripts/                      # run_table2_reconstruction, run_memory_benchmark, etc.
examples/                     # Jupyter-friendly demos
docs/
├── user/
├── dev/
├── research/
└── bench/
```
