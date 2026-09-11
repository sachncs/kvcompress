<p align="center">
  <h1 align="center">kvfold</h1>
  <p align="center">Universal plug-and-play KV cache compression for decoder-only LLMs.</p>
  <p align="center">
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License"></a>
    <a href="https://arxiv.org/abs/2607.12550"><img src="https://img.shields.io/badge/arXiv-2607.12550-red" alt="Paper"></a>
    <a href="https://github.com/sachncs/kvcompress/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/kvcompress/ci.yaml?branch=master" alt="CI"></a>
    <a href="https://sachncs.github.io/kvcompress/"><img src="https://img.shields.io/badge/docs-github%20pages-blue" alt="Docs"></a>
  </p>
</p>

Based on the paper: [*Krishnan & Schulz (2026) arXiv:2607.12550*](https://arxiv.org/abs/2607.12550).

> **Disclaimer:** I am not an author of the paper above. This repository is an independent Python re-implementation of the algorithm described in that work.

---

## Features

- **JoLT compressor:** partial Tucker decomposition on token and feature
  modes, with a JL-rotated low-bit residual; head and layer modes are
  pinned at full rank (the paper's empirical finding, Appendix B.2).
- **Flash fast variant:** randomized-SVD token mode with a context-aware
  cap `q_cap = min(max(q_min(R), ⌈T/32⌉), 512)`. 5–13× compression speedup
  at matched quality (paper §5).
- **Joint Lagrangian allocator:** decouples the global byte budget across
  cells, bisects λ to hit the target ratio. τ model `max(1 − rT/T, 1 −
  rd/d)` matches the spectral behaviour the paper measures.
- **Generic `Compressor` interface:** future compressors (quantization,
  sparsity, low-rank) plug in without touching the cache or adapter code.
- **Hugging Face integration:** transparent `DynamicCache` interception.
  Verified end-to-end on GPT-2 and Llama-family decoder-only LLMs (see
  integration tests).
- **vLLM integration:** Shape A (`export_kv` / `import_kv`) works
  everywhere; Shape B (`Offload` subclass of `vllm.v1.kv_offload`) for
  production GPU deployments.
- **Triton kernels:** fused Tucker reconstruction, JL projection, INT8
  quantize — optional, falls back to PyTorch `einsum` on non-NVIDIA systems.
- **Pure PyTorch algorithm code:** no hidden device transfer; uses
  `torch.Generator` for randomness; reversible quantization up to
  numerical noise.

---

## Installation

### From PyPI

```bash
pip install kvfold
```

### From source

```bash
git clone https://github.com/sachncs/kvcompress.git
cd kvcompress
pip install -e .
```

### Optional extras

```bash
pip install "kvfold[triton]"    # Triton kernels for reconstruction / JL
pip install "kvfold[vllm]"      # vLLM adapter
pip install "kvfold[bench]"     # matplotlib, pandas, datasets
pip install "kvfold[dev]"       # pytest, ruff, hypothesis
pip install "kvfold[docs]"      # mkdocs
```

---

## Quick Start

```python
from transformers import AutoModelForCausalLM
from kvfold import enable_compression

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
enable_compression(
    model,
    method="flash",
    target_memory="25%",     # compress KV cache to 25% of original (4×)
)

out = model.generate(...)    # KV cache is compressed transparently
```

Other methods:

```python
from kvfold import enable_compression

enable_compression(model, method="jolt", ratio=3.0)
enable_compression(model, method="flash", ratio=2.5)
enable_compression(model, method="low",   rank=128)
enable_compression(model, method="int4",  per_channel=True)
enable_compression(model, method="fp8")
enable_compression(model, method="pass")  # baseline passthrough
```

Inspect and disable:

```python
from kvfold import enable_compression

handle = enable_compression(model, method="flash", ratio=4.0)
# ... do generation ...
print(handle.stats_dict())
handle.disable()  # restore the original behaviour
```

---

## Configuration

| Setting | `enable_compression` kwarg | Default | Description |
|---|---|---|---|
| Compression method | `method` | `"flash"` | `"jolt"`, `"flash"`, `"low"`, `"int2"`, `"int4"`, `"int8"`, `"fp8"`, `"fp16"`, `"bf16"`, `"pass"` |
| Target ratio | `ratio` | — | Float > 1.0. Mutually exclusive with `target_memory`. |
| Target memory | `target_memory` | — | `"25%"` / `0.25`. Mutually exclusive with `ratio`. |
| Layer groups | `layer_groups` | `1` | Number of contiguous layer groups the allocator splits the model into. Paper uses G = 1. |
| Residual bit-widths | `bits` | `(0, 2, 4, 8)` | Tuple the allocator can choose from. |
| Cache impl name | `cache_implementation` | `"kvfold"` | Registered with HF's `cache_implementation` mechanism. |
| Seed | `seed` | `0` | Seed for randomized components (SVD, JL). |

Per-compressor kwargs (e.g. `rank=`, `factor_dtype=`, `per_channel=`,
`symmetric=`, `group_size=`) are forwarded to the chosen compressor.

---

## API

| Symbol | Module | Description |
|---|---|---|
| `Compressor` | `kvfold.core.base` | ABC for all KV cache compressors |
| `Jolt` | `kvfold.core.jolt` | Paper-faithful JoLT |
| `Flash` | `kvfold.core.flash` | Randomised-SVD JoLT variant |
| `Pass` | `kvfold.core.identity` | Passthrough baseline |
| `Low` | `kvfold.core.low` | Matrix-SVD baseline |
| `IntQuant` | `kvfold.core.int_quant` | Per-channel int quantisation baseline |
| `Bisect` | `kvfold.core.budget` | Per-cell Lagrangian allocator |
| `Greedy` | `kvfold.core.budget` | Greedy ablation baseline |
| `Pick` / `Plan` | `kvfold.core.budget` | Per-cell decision + plan dataclasses |
| `Cache` | `kvfold.store.compress` | Layer-indexed compressed cache |
| `Pool` | `kvfold.store.manager` | High-level cache facade |
| `Meta` / `LayerMeta` | `kvfold.store.metadata` | Layer-level metadata |
| `enable_compression` | `kvfold.api` | HF entry point — wraps a model's KV cache |
| `disable_compression` | `kvfold.api` | Restore the original behaviour |
| `CompressionHandle` | `kvfold.api` | Handle returned by `enable_compression` |
| `CompressionStats` | `kvfold.api` | Aggregated stats |
| `HF` | `kvfold.adapter.huggingface` | HF adapter underlying `enable_compression` |
| `Decomposer` / `Decomposition` | `kvfold.core.svd` | SVD strategy ABC + result |
| `Projector` / `Projection` | `kvfold.core.jl` | JL strategy ABC + matrix |
| `IntQuant` | `kvfold.core.quant` | Uniform int quantiser + bit-packing |
| `Residual` | `kvfold.core.residual` | JL-rotated residual payload |

---

## Examples

### End-to-end on TinyLlama

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvfold import enable_compression, CompressionHandle

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

handle: CompressionHandle = enable_compression(
    model, method="flash", target_memory="25%"
)
ids = tok("Hello, my name is", return_tensors="pt").input_ids.to(model.device)
out = model.generate(ids, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0]))

print("compression ratio:", handle.stats.compression_ratio)
handle.disable()
```

More examples in `examples/`:

- `01_quickstart.py` — minimal HF integration
- `02_direct_compression.py` — manual compress/decompress without an HF model
- `03_custom_allocator.py` — swap `Bisect` for a custom one
- `04_method_comparison.py` — JoLT vs. Low vs. int4 vs. Pass side-by-side
- `05_long_context.py` — needle-in-haystack at 8K context

---

## Error Handling

Recoverable failures emit warnings rather than raising so a single bad
layer does not halt the entire session:

- `CacheValidationError` — payload shape / dtype mismatch on `store()`.
- `AllocatorNoFeasibleError` — the per-cell candidate grid is too coarse
  for the requested ratio.
- `VLLMNotAvailableError` — guard when `vllm` is not importable.

The Hugging Face adapter replaces every blanket `except Exception` with
explicit attribute checks so structural mismatches fail loudly.

---

## Performance

Qualitative claims (paper §5; not yet measured in this re-implementation):

- **Compression time** scales with token count `T` × feature rank `rd`.
  Flash replaces the exact SVD with a randomised sketch capped at
  `q_cap = min(max(q_min(R), ⌈T/32⌉), 512)`. On Mistral-7B layer
  shapes, Flash is roughly an order of magnitude faster than exact JoLT
  at matched quality.
- **Decompress** is a single fused matmul-style kernel:
  `einsum("mar,ta,dr->mtd", core, u_token, u_feature)`. The Triton
  implementation targets ~2× vs. the PyTorch einsum on NVIDIA.
- **Bytes per layer** at 3× compression ≈ 4 × bytes_compressed of the
  Tucker core plus tiny residual payloads. See `docs/benchmarks/`.

Run the benchmarks yourself with `scripts/run_memory_benchmark.py` and
`scripts/run_speed_benchmark.py`; the published `results/` directory
contains CPU-only reference numbers from the developer's machine and is
*not* paper-reproduction data.

---

## Known limitations

- **Best called before model instantiation.** The HF adapter patches
  `transformers.DynamicCache` at call time. If a third-party module
  imports `DynamicCache` after `enable_compression` is called, that
  reference stays unpatched. Call `enable_compression` immediately
  after `from_pretrained` to avoid the race.
- **Encoder-decoder and Mamba-style models** are out of scope. The
  allocator assumes a standard decoder-only attention layout.
- **GPU benchmarks not in CI.** Reference numbers in `results/` are
  CPU-only; the perf section above is qualitative.

---

## Project Structure

```
kvfold/                            (top-level package)
  __init__.py                — Lazy-export public surface
  api.py                     — enable_compression / disable_compression / CompressionHandle
  cli.py                     — Typer CLI: version, validate, benchmark, profile, compress
  config.py                  — Typed config objects + registry
  errors.py                  — Error hierarchy
  adapter/
    base.py
    huggingface.py           — DynamicCache interception + family shim walk
    registry.py              — Per-family registry
    vllm.py                  — Shape A: export_kv / import_kv
    vllm_offload.py          — Shape B: Offload (KVCacheOffloadWorker subclass)
  core/
    base.py                  — Compressor ABC + Payload + Stats
    identity.py              — Passthrough baseline (Pass)
    low.py                   — Matrix-SVD baseline (Low)
    int_quant.py             — Int quantisation baseline (IntQuant)
    float_cast.py            — dtype-haling cast (FloatCast)
    float8.py                — E4M3/E5M2 fp8 baseline (Float8)
    quant.py                 — Int/Fp quantiser primitives + bit-packing
    residual.py              — encode/decode residual
    tucker.py                — Partial Tucker ST-HOSVD + reconstruction
    svd.py                   — Decomposer ABC (Exact + Randomized)
    jl.py                    — Johnson-Lindenstrauss projections + cache
    budget.py                — Bisect + Greedy allocators
    dispatch.py              — Compressor dispatcher + registry
    jolt.py                  — JoLT (ties everything together)
    flash.py                 — Flash (randomised mode-1)
  store/
    compress.py              — Cache
    manager.py               — Pool facade
    metadata.py              — Meta + LayerMeta
  runtime/
    pool.py                  — MemoryPool (tensor recycling)
    profile.py               — Per-call recorder
  bench/
    memory.py                — Bytes-per-method sweep
    table2.py                — Paper Table 2 reproduction
    speed.py                 — Compress / decompress wall-time
    plot.py                  — Matplotlib bar charts
  kernel/
    triton/
      tucker.py              — Fused Tucker reconstruction kernel

tests/
  unit/                      — 416 tests covering algorithms, API, contract
  property/                  — Hypothesis property tests
  integration/               — End-to-end HF tests
  regression/                — Bug-regression guards

scripts/                     — run_table2_reconstruction, run_memory_benchmark, etc.

examples/                    — 5 runnable scripts

docs/
  quickstart.md              — 30-second tour
  user/                      — API reference, performance guide, troubleshooting
  dev/                       — Architecture, contributing, testing, release
  research/                  — Algorithm notes, math reference, paper reproduction
  benchmarks/                — Running and interpreting the benchmarks
```

---

## Development

```bash
pip install -e ".[dev]"
ruff check kvfold tests examples scripts
ruff format --check kvfold tests examples scripts
mypy kvfold
pytest tests/unit tests/property         # 416 tests, ~5s on CPU
pytest --cov=kvfold --cov-report=term-missing
coverage report --fail-under=90
```

### Commit conventions

[Conventional Commits](https://www.conventionalcommits.org/) — one atomic
commit per logical impact. A milestone may have many commits.

```
feat: add q_cap auto-decoder for Flash
fix: clamp barycentric coordinates to [0,1]
docs: regenerate API reference from source
refactor: extract AABB tree to dedicated module
test: add fixtures for Worsey-Farin splits
chore: bump ruff to 0.6.x
```

---

## Testing

```bash
pytest tests/unit                       # fast (no model downloads)
pytest tests/property                   # Hypothesis property tests
pytest -m "not slow and not integration and not gpu"
pytest                                 # full suite
pytest --cov=kvfold                     # with coverage
```

The suite has skip markers: `slow`, `integration` (model downloads),
`gpu` (CUDA required). The default `pytest -m` invocation skips them.

---

## Build

```bash
pip install -e ".[docs]"                # mkdocs
mkdocs build                            # static site in site/
mkdocs serve                            # live-reload at http://127.0.0.1:8000
```

---

## Release

1. Bump version in `pyproject.toml` and `kvfold/__init__.py`.
2. Update `CHANGELOG.md` (move entries from "Unreleased" to a dated
   section).
3. Commit with `chore: release vX.Y.Z`.
4. Tag and push — CI publishes to PyPI.

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11+ |
| Deep learning | PyTorch 2.x |
| Integrations | transformers, (optional) vLLM, (optional) Triton |
| Lint / format | ruff (`ruff check`, `ruff format`) |
| Type check | mypy (configured; missing-import-permissive) |
| Tests | pytest, Hypothesis |
| Coverage | coverage.py |

---

## Roadmap

### High priority

- **GPU Triton path verification on a CUDA box:** the PyTorch fallback is
  verified; only the Triton kernel needs an end-to-end run on real
  hardware to confirm the 2× speedup claim.
- **Vector-valued higher-order projections:** not applicable to JoLT, but
  the registry is wired to accept them once they exist.

### Medium priority

- **vLLM Shape C — custom attention backend:** the third integration
  shape, currently stubbed in `adapter/vllm.py`.
- **`vllm 0.x` → `vllm 1.0` API migration:** the
  `vllm.v1.kv_offload.base.KVCacheOffloadWorker` superclass is stable
  across `1.0` releases but the import path may move.

### Low priority / research

- **Learned projector:** replace the JL projection with a learned
  projection trained on a calibration set; would only help if the
  ε²(b) calibration curve has high variance across layers.
- **Per-layer allocator calibration:** ship `tau_table` lookup tables
  for the popular architectures (Llama-3 8B, Mistral-7B, Qwen2.5-7B)
  so the allocator's τ error model is exact rather than monotone.

---

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct and the process for submitting pull requests.

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

## Security

Report vulnerabilities to **sachncs@gmail.com** — see
[SECURITY.md](SECURITY.md).

## License

[Apache-2.0](LICENSE) © 2026 sachin.
