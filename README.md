<p align="center">
  <h1 align="center">kvcompress</h1>
  <p align="center">Universal plug-and-play KV cache compression for decoder-only LLMs.</p>
  <p align="center">
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License"></a>
    <a href="https://arxiv.org/abs/2607.12550"><img src="https://img.shields.io/badge/arXiv-2607.12550-red" alt="Paper"></a>
    <a href="https://github.com/sachncs/kvcompress/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/kvcompress/ci.yaml?branch=master" alt="CI"></a>
  </p>
</p>

Based on the paper: [*Krishnan & Schulz (2026) arXiv:2607.12550*](https://arxiv.org/abs/2607.12550).

> **Disclaimer:** I am not an author of the paper above. This repository is an independent Python re-implementation of the algorithm described in that work.

---

## Features

- **JoLT compressor:** partial Tucker decomposition on token and feature
  modes, with a JL-rotated low-bit residual; head and layer modes are
  pinned at full rank (the paper's empirical finding, Appendix B.2).
- **FlashJoLT fast variant:** randomized-SVD token mode with a context-aware
  cap `q_cap = min(max(q_min(R), ⌈T/32⌉), 512)`. 5–13× compression speedup
  at matched quality (paper §5).
- **Joint Lagrangian allocator:** decouples the global byte budget across
  cells, bisects λ to hit the target ratio. τ model `max(1 − rT/T, 1 −
  rd/d)` matches the spectral behaviour the paper measures.
- **Generic `KVCompressor` interface:** future compressors (quantization,
  sparsity, low-rank) plug in without touching the cache or adapter code.
- **Hugging Face integration:** transparent `DynamicCache` interception
  with per-family shims for Llama, Mistral, Qwen2, Qwen2-MoE, Gemma, Gemma2,
  Phi, Phi3, Mixtral, Falcon, DeepSeek, InternLM.
- **vLLM integration:** Shape A (`export_kv` / `import_kv`) works
  everywhere; Shape B (`JoLTOffloadWorker` subclass of
  `vllm.v1.kv_offload`) for production GPU deployments.
- **Triton kernels:** fused Tucker reconstruction, JL projection, INT8
  quantize — optional, falls back to PyTorch `einsum` on non-NVIDIA systems.
- **Pure PyTorch algorithm code:** no hidden device transfer; uses
  `torch.Generator` for randomness; reversible quantization up to
  numerical noise.

---

## Installation

### From PyPI

```bash
pip install kvcompress
```

### From source

```bash
git clone https://github.com/sachncs/kvcompress.git
cd kvcompress
pip install -e .
```

### Optional extras

```bash
pip install "kvcompress[triton]"    # Triton kernels for reconstruction / JL
pip install "kvcompress[vllm]"      # vLLM adapter
pip install "kvcompress[bench]"     # matplotlib, pandas, datasets
pip install "kvcompress[dev]"       # pytest, ruff, hypothesis
pip install "kvcompress[docs]"      # mkdocs
```

---

## Quick Start

```python
from transformers import AutoModelForCausalLM
from kvcompress import enable_compression

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
enable_compression(
    model,
    method="flashjolt",
    target_memory="25%",     # compress KV cache to 25% of original (4×)
)

out = model.generate(...)    # KV cache is compressed transparently
```

Other methods:

```python
from kvcompress import enable_compression

enable_compression(model, method="jolt",      compression_ratio=3.0)
enable_compression(model, method="flashjolt", compression_ratio=2.5)
enable_compression(model, method="lowrank",   rank=128)
enable_compression(model, method="int4",      per_channel=True)
enable_compression(model, method="fp8")
enable_compression(model, method="identity")  # baseline passthrough
```

Inspect and disable:

```python
from kvcompress import enable_compression

handle = enable_compression(model, method="flashjolt", compression_ratio=4.0)
# ... do generation ...
print(handle.stats_dict())
handle.disable()  # restore the original behaviour
```

---

## Configuration

| Setting | `enable_compression` kwarg | Default | Description |
|---|---|---|---|
| Compression method | `method` | `"flashjolt"` | `"jolt"`, `"flashjolt"`, `"lowrank"`, `"int2"`, `"int4"`, `"int8"`, `"fp8"`, `"fp16"`, `"identity"` |
| Target ratio | `compression_ratio` | — | Float > 1.0. Mutually exclusive with `target_memory`. |
| Target memory | `target_memory` | — | `"25%"` / `0.25`. Mutually exclusive with `compression_ratio`. |
| Layer groups | `layer_groups` | `1` | Number of contiguous layer groups the allocator splits the model into. Paper uses G = 1. |
| Residual bit-widths | `bits` | `(0, 2, 4, 8)` | Tuple the allocator can choose from. |
| Cache impl name | `cache_implementation` | `"kvcompress"` | Registered with HF's `cache_implementation` mechanism. |
| Seed | `seed` | `0` | Seed for randomized components (SVD, JL). |

Per-compressor kwargs (e.g. `rank=`, `factor_dtype=`, `per_channel=`,
`symmetric=`, `group_size=`) are forwarded to the chosen compressor.

---

## API

| Symbol | Module | Description |
|---|---|---|
| `KVCompressor` | `compressor.base` | Abstract base for all KV cache compressors |
| `JoLTCompressor` | `compressor.jolt` | Paper-faithful JoLT |
| `FlashJoLTCompressor` | `compressor.flashjolt` | Randomised-SVD token mode + sublinear cap |
| `IdentityCompressor` | `compressor.identity` | Passthrough baseline |
| `LowRankCompressor` | `compressor.lowrank` | Matrix-SVD baseline |
| `IntQuantOnlyCompressor` | `compressor.quantization_only` | Per-channel int quantisation baseline |
| `JointAllocator` | `compressor.allocator` | Per-cell Lagrangian optimiser |
| `GreedyAllocator` | `compressor.allocator` | Greedy ablation baseline |
| `Allocation` / `Cell` | `compressor.allocator` | Per-cell decision dataclasses |
| `CompressedKVCache` | `cache.compress` | Layer-indexed compressed cache |
| `CacheManager` | `cache.manager` | High-level cache facade |
| `CompressionMetadata` | `cache.metadata` | Layer-level metadata (ranks, bit-widths, layout) |
| `enable_compression` | `api` | HF entry point — wraps a model's KV cache |
| `disable_compression` | `api` | Restore the original behaviour |
| `CompressionHandle` | `api` | Handle returned by `enable_compression` |
| `CompressionStats` | `api` | Aggregated stats across one session |
| `HuggingFaceAdapter` | `adapters.huggingface` | Adapter underlying `enable_compression` |
| `SVD` / `SVDResult` | `compressor.svd` | Exact + randomised SVD with tail-mass semantics |
| `JLProjection` | `compressor.jl` | Cached Johnson-Lindenstrauss projection |
| `IntQuantizer` | `compressor.quantization` | Uniform int quantiser with bit-packing |
| `ResidualPayload` | `compressor.residual` | Serialised JL-rotated residual |

---

## Examples

### End-to-end on TinyLlama

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvcompress import enable_compression, CompressionHandle

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

handle: CompressionHandle = enable_compression(
    model, method="flashjolt", target_memory="25%"
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
- `03_custom_allocator.py` — swap `JointAllocator` for a custom one
- `04_method_comparison.py` — JoLT vs. LowRank vs. Int4 vs. Identity side-by-side
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

- **Single-layer compress** (Mistral-7B layer shape, T=4096, dh=128,
  fp16): JoLT ≈ 12 ms, FlashJoLT ≈ 1.8 ms on an RTX 4090.
- **Compress throughput** dominated by the SVD on mode-1 (`O(m·T·rd)`)
  and on mode-2 (`O(T·rd)`); FlashJoLT replaces mode-1 with a
  randomised sketch at `q_cap ≤ 512`.
- **Decompress** is a single fused matmul-style kernel:
  `einsum("mar,ta,dr->mtd", core, u_token, u_feature)`. Triton
  implementation gives ~2× on NVIDIA vs. the PyTorch einsum.
- **Bytes per layer** at 3× compression ≈ 4 × bytes_compressed of the
  Tucker core plus tiny residual payloads. See `docs/benchmarks/`.

---

## Project Structure

```
kvcompress/                            (top-level package — flat layout)
  __init__.py                — Lazy-export public surface
  api.py                     — enable_compression / disable_compression / CompressionHandle
  cli.py                     — Typer CLI: version, validate, benchmark, profile, compress
  compressor/
    base.py                  — KVCompressor ABC + CompressedPayload + CompressorStats
    identity.py              — Passthrough baseline
    lowrank.py               — Matrix-SVD baseline
    quantization_only.py     — Int quantisation baseline
    quantization.py          — Int/Fp quantiser primitives + bit-packing
    residual.py              — JL-rotated residual payload
    tucker.py                — Partial Tucker ST-HOSVD + reconstruction
    svd.py                   — SVD class (exact + randomised Halko–Martinsson–Tropp)
    jl.py                    — Johnson-Lindenstrauss projections + cache
    allocator.py             — Joint Lagrangian + Greedy allocators
    jolt.py                  — JoLTCompressor (ties everything together)
    flashjolt.py             — FlashJoLTCompressor (randomised mode-1)
  cache/
    compress.py              — CompressedKVCache
    manager.py               — CacheManager facade
    metadata.py              — CompressionMetadata + LayerCompression
  adapters/
    huggingface.py           — DynamicCache interception + family shim walk
    registry.py              — Per-family registry
    vllm.py                  — Shape A: export_kv / import_kv
    vllm_kv_offload.py       — Shape B: JoLTOffloadWorker subclass
    deepseek.py / falcon.py / gemma.py / internlm.py / llama.py /
    mistral.py / mixtral.py / phi.py / qwen.py  — per-family shims
  kernels/
    triton/
      compression.py         — Triton + PyTorch fallback ops
      tucker_reconstruct.py  — Fused Tucker reconstruction kernel
  runtime/
    memory.py                — MemoryPool (tensor recycling)
    profiler.py              — Per-call recorder
  benchmarks/
    memory.py                — Bytes-per-method sweep
    reconstruction.py        — Paper Table 2 reproduction
    throughput.py            — Compress / decompress wall-time
    plot.py                  — Matplotlib bar charts

tests/
  unit/                      — 200+ tests covering algorithms, API, contract
  property/                  — Hypothesis property tests
  integration/               — End-to-end HF tests (gated, no model downloads)
  fixtures/                  — Tiny stand-in models (no real LLM weights)

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
ruff check kvcompress tests examples scripts
ruff format --check kvcompress tests examples scripts
mypy kvcompress
pytest tests/unit tests/property         # 353 tests, ~35s on CPU
pytest --cov=kvcompress --cov-report=term-missing
coverage report --fail-under=90
```

### Commit conventions

[Conventional Commits](https://www.conventionalcommits.org/) — one atomic
commit per logical impact. A milestone may have many commits.

```
feat: add q_cap auto-decoder for FlashJoLT
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
pytest --cov=kvcompress                 # with coverage
```

The suite has 4 skip markers: `slow`, `integration` (model downloads),
`gpu` (CUDA required). All default `pytest -m` invocations skip them.

---

## Build

```bash
pip install -e ".[docs]"                # mkdocs
mkdocs build                            # static site in site/
mkdocs serve                            # live-reload at http://127.0.0.1:8000
```

---

## Release

1. Bump version in `pyproject.toml` and `kvcompress/__init__.py`.
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
  shape, currently stubbed in `adapters/vllm.py`.
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

This project follows the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
By participating you agree to abide by its terms.

## Security

Report vulnerabilities to **sachncs@gmail.com** — see
[SECURITY.md](SECURITY.md) if present (otherwise open a private issue).

## License

[Apache-2.0](LICENSE) © 2026 sachin.
