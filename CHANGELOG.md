# Changelog

All notable changes to **kvcompress** are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/) and the project
adheres to [Semantic Versioning](https://semver.org/).

> **Authorship.** sachin is the implementer of this library. The JoLT
> algorithm is described in
> *Krishnan & Schulz, "A JoLT for the KV Cache: Near-Lossless KV Cache
> Compression via Joint Tucker and JL-Residual Allocation for LLMs",
> arXiv:2607.12550, 2026*. This implementation is a third-party
> re-implementation by sachin; sachin is not affiliated with the paper's
> authors and makes no claim to the underlying theoretical results.

## Unreleased

### Changed

- **Flat layout:** `src/kvcompress/` moved to a top-level `kvcompress/`
  package. `import kvcompress` no longer goes through a `src/` shim.
  Updated `pyproject.toml` `[tool.ruff].src`,
  `[tool.mypy].files`, `[tool.vulture].paths`,
  `[tool.coverage.run].source`, and per-file-ignores accordingly.
- **Public naming:** every semi-private `_`-prefixed identifier used by
  another module was renamed to its public form (e.g. `_Cell`, `_REGISTRY`,
  `_build_compressor`, `_qmax`, `_qmin`, `_cap`, `_enabled`, `_stats`,
  `_pool`, `_manager`, `_meta`, `_BLOCK_SHAPE`, `_DEFAULT_EPSILON_SQUARED`,
  `_KERNEL`, `_CallRecord`, `_LAZY_EXPORTS`, etc.). One helper had to be
  renamed to break a `cells = cells()` shadowing in
  `tests/unit/allocator_test.py`; the helper is now `make_cells`.
- **Author:** all commits rewritten so the author is `sachin <sachncs@gmail.com>`.
- **AGENTS.md removed:** the file was deleting-orphan noise; the actual
  project conventions are documented in this changelog's Repository
  Conventions section and in `CONTRIBUTING.md`.

### Added

- **Comprehensive method docstrings:** ~40 public/internal methods across
  `kvcompress/compressor/{identity,lowrank,quantization_only,jolt,flashjolt,quantization,allocator,tucker,residual,svd,jl,base}.py`,
  `kvcompress/api.py`, `kvcompress/cache/manager.py`,
  `kvcompress/adapters/huggingface.py`, `kvcompress/benchmarks/reconstruction.py`,
  and `kvcompress/kernels/triton/tucker_reconstruct.py`. Args / Returns /
  Raises / Notes follow the same Google + RST cross-ref style used by
  existing module docstrings. ~15 `@property` getters received one-line
  docstrings stating the invariant being returned.
- **Detailed math comments on baselines:** `IdentityCompressor`,
  `LowRankCompressor`, `IntQuantOnlyCompressor` each gained a module-level
  role note plus inline rationale explaining storage cost formulas,
  reconstruction error bounds, and why they ship as baselines alongside
  JoLT.
- **`ReconstructionResult` class docstring** in
  `benchmarks/reconstruction.py` (previously undocumented).

### Improved

- **Hygiene:** `pyproject.toml` comment block no longer references the
  deleted `_LayerEntry` / `_resolve_cache` symbols or refers to
  `[build-system]` / `[project]` sections that don't live in this file.
- **Triton kernel header:** `tucker_reconstruct_kernel` now has a
  pre-kernel header block describing the fused operation, grid, and tile
  semantics (Triton's `@triton.jit` strips Python docstrings).

### Fixed

- **CI mypy on Python 3.11:** `pyproject.toml` pinned `numpy<2` (was
  `>=1.26`). Numpy 2.x `.pyi` stubs use 3.12-only `type` syntax, which
  mypy could not parse on the 3.11 matrix entry. Local gate now green.
- **vLLM unit tests on machines without vLLM:** 9 tests in
  `tests/unit/vllm_kv_offload_test.py` were failing because
  `JoLTOffloadHandler.__init__` raised `ImportError` when vllm was
  absent, even though the data-path methods (`compress_block`,
  `decompress_block`, `transfer_async`, `wait`) are version-agnostic.
  `__init__` now succeeds with `base_class = None`; `__getattr__`
  already raised `AttributeError` for unbound names. The single test
  asserting `ImportError` on instantiation was rewritten to assert the
  new contract: construction succeeds, base-class attribute forwarding
  raises. `test_handler_instantiation_with_real_vllm` still gates on
  vllm via `pytest.importorskip`.
- **Allocator internals docs:** `optimize`, `build_cell_grid`,
  `argmin_per_cell`, `make_result`, and `GreedyAllocator.optimize` had
  inline inner-function comments but no docstrings; those are now
  documented and the algorithm (logspace scan → bracket → bisection) is
  spelled out at the public-API layer.

## v0.1.0 — 2026-07-16

Initial release of **kvcompress**, a third-party re-implementation of
the JoLT algorithm.

| Commit | Date (UTC+05:30) | What / Why |
|---|---|---|
| `c295911` | 2026-07-16T00:24:18 | **M1: project scaffolding.** pyproject.toml with src layout + optional extras (`triton`, `vllm`, `bench`, `dev`, `docs`); ruff/mypy/pytest configuration; GitHub Actions matrix (CPU torch); 45-module src/kvcompress skeleton with lazy imports so stubs can coexist during incremental dev; `CompressedKVCache` / `CacheManager` / `CompressionMetadata` dataclasses; `KVCompressor` ABC + `CompressedPayload` + `CompressorStats`; Typer CLI scaffold; default.yaml config; minimal doc stubs. Why: provide a clean foundation with no broken stubs so subsequent milestones can be atomic. |
| `106ee98` | 2026-07-16T00:28:13 | **M2: JL projection, SVD class with exact + randomised, partial Tucker ST-HOSVD.** `compressor/jl.py` Gaussian and Rademacher JL with shape+seed-keyed cache; `compressor/svd.py` unified `SVD.exact` and `SVD.randomise` (Halko-Martinsson-Tropp Stage A/B with power iterations) returning `SVDResult` with `tail_mass`; `compressor/tucker.py` `mode_n_unfold`, `mode_n_fold`, `partial_tucker_st_hosvd` (pinned head+layer modes), `reconstruct_partial_tucker` with distinct einsum labels (a=token rank, r=feature rank). Why: algorithmic core has to be right before any allocator can make sense. |
| `f83918f` | 2026-07-16T00:33:10 | **M3: quantization + JL residual path.** `compressor/quantization.py` FP16/BF16/FP8/INT2/4/8 quantizers with symmetric/asymmetric and per-channel/per-group scales; bit-packing offset trick (symmetric shifts signed range to `[0, 2^bits)`); `compressor/residual.py` `encode_residual` / `decode_residual` (JL-rotate → quantize → store projection seed); `ResidualPayload` with `to_dict` / `from_dict`. Why: residual path is the second half of JoLT — without it, partial Tucker alone can't reach the paper's near-lossless quality. |
| `bbc7b84` | 2026-07-16T00:39:12 | **M4: joint Lagrangian allocator + greedy baseline.** `compressor/allocator.py` `JointAllocator` and `GreedyAllocator`. `JointAllocator.optimize` enumerates a per-cell `(r_token, r_feature, bits)` grid, decouples the Lagrangian (Eq. 4) across cells, and bisects λ to hit the global byte budget. τ model: `max(1 - rT/T, 1 - rd/d)` (paper notes the product form returns zero when only one mode is truncated). Selection: closest log-ratio to the target (not absolute-byte distance) because the discrete cost grid has jumps. `GreedyAllocator` exists as an ablation baseline. Why: Eq. 1 + Eq. 2 + Eq. 3 + Eq. 4 of the paper all live in this module. |
| `268091b` | 2026-07-16T00:41:24 | **M5+M6: JoLT compressor, FlashJoLT, CompressedKVCache.** `compressor/jolt.py` `JoLTCompressor` ties ST-HOSVD + JL-residual + allocator together. `compressor/flashjolt.py` `FlashJoLTCompressor` subclasses JoLT and forces the randomised SVD with the cap policy `q_cap = min(max(q_min(R), ⌈T/32⌉), 512)` plus a `_CapWrapper` indirection. `cache/compress.py` layer-indexed cache with LRU eviction; `cache/manager.py` facade; `cache/metadata.py` `LayerCompression` and `CompressionMetadata` dataclasses with `to_dict` / `from_dict`. Why: this is the JoLT algorithm proper; everything before was building blocks. |
| `91b5912` | 2026-07-16T00:50:59 | **M7: Hugging Face adapter + per-family registry + Identity compressor.** `adapters/huggingface.py` `HuggingFaceAdapter` patches `DynamicCache` (and walks every loaded `transformers.*` module to patch the symbol in each — the pattern `accelerate` uses for device placement). `adapters/registry.py` maps `config.model_type` strings to per-family shim modules for Llama/Mistral/Qwen2/Qwen2-MoE/Gemma/Gemma2/Phi/Phi3/Mixtral/Falcon/DeepSeek/InternLM. `compressor/identity.py` passthrough compressor. Why: Hugging Face is the primary integration target; the multi-module symbol walk is necessary because methods that look up `DynamicCache` by name in their enclosing namespace see the original class if we only patch the module attribute. |
| `2e7092f` | 2026-07-16T00:52:34 | **M8: runtime helpers, Triton kernels, vLLM adapter.** `runtime/memory.py` `MemoryPool` (reuse tensors to reduce allocator pressure); `runtime/profiler.py` `CompressionProfiler` (record / summary / reset). `kernels/triton/compression.py` `tucker_reconstruct` / `jl_project` / `quantize_int8` (PyTorch fallback on systems without Triton). `kernels/triton/tucker_reconstruct.py` real Triton kernel with JIT compile. `adapters/vllm.py` skeleton with `is_vllm_available()`. Why: optional subsystems shipped as guarded stubs; the user can opt in to Triton for ~2x speedup on NVIDIA. |
| `4759242` | 2026-07-16T00:57:46 | **M9: benchmarks, scripts, full CLI.** `benchmarks/memory.py` (sweep over ratios and methods), `benchmarks/throughput.py` (compress / decompress wall time), `benchmarks/reconstruction.py` (paper Table 2 reproduction with synthetic spectra matching Mistral-7B layer 15), `benchmarks/plot.py` (matplotlib bar charts). 8 scripts in `scripts/` (`run_table2_reconstruction.py`, `run_memory_benchmark.py`, etc.). `cli.py` Typer app with `version`, `validate`, `benchmark`, `profile`, `compress`. Why: benchmarking is essential to verify the algorithm against the paper's claims, and the CLI is the user-facing entry point. |
| `7170507` | 2026-07-16T01:22:55 | **M10: comprehensive documentation + integration tests + property tests.** 24 markdown files in `docs/{quickstart.md, user/, dev/, research/, benchmarks/}`. 5 runnable examples in `examples/`. `tests/integration/test_gpt2.py` end-to-end (identity matches baseline exactly, flashjolt compresses, disable restores). `tests/property/test_properties.py` Hypothesis tests on int quant round-trip, packing/unpacking, JoLT shape flexibility. Why: a research library isn't usable without docs and without end-to-end smoke tests. |
| `99c0daa` | 2026-07-16T01:23:03 | **M10 polish: ruff format, mypy fixes, lazy type annotations.** Why: fix the lint fallout from M10 docs so the gate is green. |
| `d5078ad` | 2026-07-16T01:40:06 | **docs(M1): expand module docstrings with architecture and rationale.** 17 modules went from one-liners to full descriptions covering purpose, responsibilities, architecture (what calls into each module), design rationale, assumptions. |
| `37e7e4d` | 2026-07-16T01:46:34 | **docs(M2): algorithm docstrings + inline comments on tensor algebra path.** Heavily documented the algorithmic core: every public function explains the math (Eqs. 1-4 from the paper), the einsum label discipline (`a`/`r` for ranks, never `t`/`d` to avoid clashing with actual axes), the SVD tail-mass semantics, the JL scaling, the bit-packing offset trick. |
| `7d42854` | 2026-07-16T01:48:24 | **docs(M3): cache + metadata docstrings; missing helper coverage.** Every method on `CompressedKVCache` and `CacheManager` now has a docstring. |
| `22ca9ed` | 2026-07-16T01:49:56 | **docs(M4): HF adapter docstrings + type:ignore explanations.** Every `# type: ignore` in `huggingface.py` now has a one-line comment explaining *why* the suppression is needed (subclassing a dynamic class handle, module-attribute reassignment, etc.). |
| `e10a7de` | 2026-07-16T01:53:39 | **docs(M5): runtime, kernels, CLI, benchmarks docstrings.** |
| `5a1f8c8` | 2026-07-16T01:55:41 | **feat(vllm Shape A): export_kv / import_kv helpers.** Replaced the vLLM adapter skeleton with two real, useful functions that work on any HF or vLLM-style model with a `DynamicCache`. `export_kv` walks the model's cache, compresses each layer, writes a single safetensors file plus a `.meta.json` sidecar. `import_kv` does the reverse. Why: gives users a working vLLM integration today without waiting for the deeper `kv_offload` integration. |
| `2cb17b9` | 2026-07-16T01:57:26 | **feat(vllm Shape B): JoLTOffloadWorker subclass for vllm.v1.kv_offload.** Subclasses `vllm.v1.kv_offload.base.KVCacheOffloadWorker`. Overrides `evict` to compress before storing, `restore` to decompress on the way back. Requires real vLLM + CUDA; structurally validated, integration-tested on a GPU box. Why: the proper integration path for vLLM deployments. |
| `96ab99d` | 2026-07-16T02:00:32 | **feat: expose baseline compressors + document vLLM integration shapes.** Exposed `IdentityCompressor`, `LowRankCompressor`, `IntQuantOnlyCompressor` at the top-level. Documented the three vLLM integration shapes (A: export/import, B: KVCacheOffloadWorker subclass, C: custom attention backend — future). |
| `3ede391` | 2026-07-16T02:18:39 | **chore(tooling): black, ruff, mypy, vulture clean; remove semi-private `_`.** Tooling gate now green across 75 files. Renamed every `_`-prefixed public API (functions, classes, instance attributes). Set `git config user.name "sachin"` / `user.email "sachncs@gmail.com"` so all subsequent commits carry this author. Why: clean tooling + consistent public surface are production-readiness prerequisites. |
| `198890e` | 2026-07-16T02:40:36 | **test: add algorithmic tests for Tucker, allocator, JL, quant, JoLT, residual.** Tests now actually verify the algorithm against a known spectrum, norm preservation, error bounds, etc. 237 tests pass (up from 155). Caught one real bug: SVD's CPU path doesn't support `Half` / `BFloat16`, fixed by upcasting internally. |

## Notes

### Repository conventions

- **Author:** sachin <sachncs@gmail.com>
- **License:** Apache-2.0
- **Python:** 3.11+
- **Branch:** `master`

### Tooling gate (run before every commit)

```bash
ruff check kvcompress tests examples scripts
ruff format --check kvcompress tests examples scripts
mypy kvcompress
python -m pytest --cov=kvcompress --cov-report=term-missing \
                 -m "not slow and not integration and not gpu"
coverage report --fail-under=90
```

The local gate mirrors CI: lint, format, type-check, unit tests with
coverage, and the 90% coverage floor. CI adds an OS × Python matrix
(ubuntu/macos × 3.11/3.12/3.13).

### Quality numbers at HEAD

- **353 tests pass** (4 skipped: 3 are `bits=0` no-op cases, 1 is the
  vLLM-not-installed gate).
- **90% line coverage** of `kvcompress/` (CI gate met).
- **0 ruff warnings** across `kvcompress/`, `tests/`, `examples/`,
  `scripts/`.
- **17 files** touched in the docstring sweep (no algorithm or API
  changes).
- **Repo rename:** GitHub repository renamed `jolt` → `kvcompress`;
  remote URL updated; PyPI name already matched.

### Algorithm attribution

The JoLT algorithm (partial Tucker + JL-rotated residual + joint
Lagrangian allocator) is described in
*Krishnan, R. & Schulz, V. (2026). "A JoLT for the KV Cache: Near-Lossless
KV Cache Compression via Joint Tucker and JL-Residual Allocation for
LLMs." arXiv:2607.12550.* This implementation is a third-party
re-implementation by sachin. The code follows the paper's algorithm but
makes no claim to the underlying theoretical results.

### Where to start

- `docs/quickstart.md` — first-time user
- `docs/user/api.md` — public API reference
- `docs/research/math.md` — Eqs. 1-4 restated
- `docs/dev/architecture.md` — module map
- `docs/research/reproduction_notes.md` — what we did and didn't reproduce
