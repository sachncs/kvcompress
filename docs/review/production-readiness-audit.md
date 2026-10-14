# kvfold production-readiness audit

Date: 2026-10-14

This is a repository and live-site review, not a claim that the project is ready
for production. The findings below are intentionally broader than a top-ten list.
The original baseline was captured before remediation; the current verification
status is recorded below.

## Verification baseline

| Check | Result | Meaning |
| --- | --- | --- |
| `uv run python -m compileall -q kvfold tests scripts` | Pass | Python files compile in the pinned local environment. |
| `uv run pytest -m 'not slow and not integration and not gpu'` | Pass | 454 passed, 4 expected skips, 18 deselected under Python 3.12.14; 90.97% coverage after removing dead compatibility tests. |
| `uv run pytest -m integration -q` | Pass | 4 pinned GPT-2 tests, 10 MPS tests, and 2 CPU/MPS cache-residency tests passed locally; 460 tests deselected. |
| `uv run pytest -m mps -q` | Pass | 10 Apple MPS tests passed, including Flash's CPU fallback for the MPS-missing QR operator; FP8 is explicitly tested as unsupported. |
| `uv run coverage report --fail-under=90` | Pass | 90.97% on the core package; optional CUDA/Triton, vLLM, and benchmark/research paths remain separate. |
| `uv run mypy kvfold` | Pass | No issues in 52 source files. |
| `uv run pydocstyle --convention=google ... kvfold` | Pass | Public classes, methods, and functions pass the documented Google-style check; module/package/constructor exceptions are explicit. |
| `uv run vulture kvfold tests examples scripts` | Pass | No unused-code findings. |
| `uv run ruff check kvfold tests examples scripts` | Pass | Intentional lazy-import/scientific-notation exceptions are documented in `ruff.toml`; no remaining findings. |
| `site/npm run build` | Pass | Vite production build succeeds. |
| `site/npm run typecheck` | Pass | TypeScript project references type-check. |
| `site/npm run lint` | Pass | ESLint 10 flat config passes after the toolchain upgrade. |
| `site/npm audit` | Pass | No production or development dependency advisories reported after the Vite 8/Tailwind 4 migration. |
| `site/npm run test:e2e` | Pass | Five Chromium smoke tests cover navigation, mobile menu, theme, external evidence links, FAQ semantics, and reduced motion. |
| `uv lock --check` | Pass | Locked resolution includes the optional vLLM range and NumPy 2-compatible dependency constraints. |
| `uv build` + `uvx twine check dist/*` | Pass | Source distribution and universal wheel build successfully; both artifacts pass metadata validation and the wheel contains `kvfold/`. |
| `scripts/validate_install.py` static gate | Pass | Smoke-test failure paths now return non-zero explicitly instead of relying on `assert`, which can be disabled with optimized Python. |
| `uv run python scripts/check_doc_links.py` | Pass | All relative Markdown links resolve from the repository root. |
| `.github/workflows/{ci,release,gpu}.yaml` parse | Pass | Release and manual GPU workflow definitions also parse successfully with PyYAML. |

## Principal-engineer verdict

The codebase is materially cleaner after this pass: compatibility-only public
aliases, the no-op family registry, the process-wide Transformers fallback, and
their tests/docs were removed. CPU, MPS, GPT-2 integration, static analysis, and
frontend gates are reproducible locally.

The storage architecture is now reliable for the validated CPU/MPS path: the HF
adapter keeps framework layer objects as metadata only and stores K/V payloads
in the compressed Pool. Authoritative CPU/MPS tests prove that framework-backed
K/V storage is absent and compressed tensor storage is below raw K/V storage.
This supports a production-readiness claim for the tested CPU/MPS scope, not a
universal claim. CUDA/Triton execution, real vLLM, long-context stress, full
device/dtype coverage, clean external CI, and live GitHub Pages verification
remain release gates; CUDA is explicitly pending.

## Findings

### P0 — release blockers

1. **Resolved for the local/CPU path in this pass:** `uv.lock`, `uv sync --extra dev`,
   and documented `uv run` commands reproduce the Python gate. CI now uploads
   Python versions and frozen dependency manifests for every matrix cell.
2. **Resolved for the fast gate in this pass:** the enforced core threshold is
   90%, and stale test-count/coverage instructions were removed or qualified.
3. **Resolved in the local source, not yet deployed:** the site copy now uses
   scoped evidence-led claims and the CI/browser gates are documented. The live
   GitHub Pages URL must still be republished and checked before the old claims
   can be considered removed from the public surface.
4. **Resolved for supported Transformers models in this pass:** current models
   use a model-scoped `_prepare_cache_for_generation` hook, verified by GPT-2
   integration tests; unsupported model objects without that hook now fail
   clearly instead of entering a process-wide fallback path.
5. **Resolved for the canonical package surface in this pass:** the package is
   `kvfold`, the product is `kvfold`, and the repository slug is `kvcompress`;
   compatibility aliases and dead adapter shim modules were removed. A final
   external-link sweep is still required.

### P1 — reliability and correctness

6. **Resolved in this pass:** rollback now catches ordinary `Exception` values,
   preserving process-control exceptions such as `KeyboardInterrupt` and
   `SystemExit`.
7. **Resolved in this pass:** adapter enable/disable transitions are serialized
   by a process-wide re-entrant lock, and the only supported integration path is
   the model-scoped Transformers cache hook. The process-wide fallback patcher
   was removed.
8. **Resolved in this pass:** `INSTALLED_ADAPTERS` uses a weak-key mapping and
   model-scoped adapters can coexist without mutating global Transformers module
   symbols.
9. **Resolved in this pass:** `cache_implementation` accepts only
   `dynamic`/`dynamic_full`; the `kvfold` compatibility alias was removed and
   invalid values fail during setup.
10. **Resolved in this pass:** unknown per-method kwargs now raise
    `MethodConfigError` instead of being silently discarded.
11. **Resolved for the in-memory cache in this pass:** replacement, eviction,
    metadata-index rebuilding, LRU reads, pool live-layer synchronization, and
    failed metadata preparation now preserve explicit invariants with regression
    tests. Persistence-level transactionality remains separate work.
12. **Resolved for CPU/MPS in this pass:** `HF.KvCompressCache.update()` no
    longer calls the parent implementation that retains full K/V tensors. The
    framework layer objects remain metadata-only, while the compressed `Pool`
    is the persistent backing store; reconstruction is transient for attention.
    CPU and MPS residency tests assert this directly and compare backing tensor
    storage with raw K/V bytes. CUDA allocator fragmentation, metadata overhead,
    and process-level reserved memory still require a CUDA deployment run.
13. **Partially resolved in this pass:** CPU float16/bfloat16/float8 paths,
    singleton and zero-length contracts, and several edge dimensions are tested;
    CUDA, non-contiguous/long-context stress, and the full dtype/device matrix
    remain release-critical dimensions.
14. **Partially resolved in this pass:** the vLLM extra is now `>=0.19,<1.0`,
    the lockfile resolves it with NumPy 2, and the handler validates required
    methods/version before use. Real vLLM execution remains unverified locally.
15. **Improved in this pass, hardware verification still pending:** a manual
    self-hosted CUDA/Triton workflow now runs the parity suite and records GPU,
    CUDA, and PyTorch versions. It cannot be treated as passing until a real
    GPU runner executes it; performance claims remain provisional.
16. **Partially resolved in this pass:** cache and Pool state are now serialized
    and concurrent store/retrieve regression coverage exists. A seed/device/
    thread-count reproducibility contract is documented; cross-device and
    cross-PyTorch equivalence still needs a matrix test.
17. **Partially resolved in this pass:** a typed error/recovery policy is now
    documented, rollback avoids `BaseException`, and core errors carry stable
    codes/hints. Remaining work is to audit every warning/catch site and attach
    layer/shape/dtype/device context consistently.

### P1 — testing and release engineering

18. **Mostly resolved in this pass:** the four GPT-2 network integration
    tests pass locally, and CI now runs them on a scheduled or manually
    dispatched integration job. The GPT-2 revision is now pinned; cached model
    artifacts and release provenance are still required. The vLLM hardware workflow now points at the actual
    offload test module, but cannot run locally without vLLM/CUDA.
19. **CPU and MPS validated; CUDA/Triton remains pending:** the manual
    `.github/workflows/gpu.yaml` self-hosted lane is defined, but this Apple
    Silicon environment has no CUDA device. A successful CUDA/Triton hardware
    run is still required before GPU release claims can be upgraded.
20. **Partially resolved in this pass:** multi-model isolation, rollback,
    repeated disable, third-party symbol mutation, cache lifecycle races, and
    concurrent cache operations now have regression coverage. Import-order and
    process-shutdown tests remain.
21. **Partially resolved in this pass:** a compatibility matrix is documented,
    Python and Apple MPS are covered locally, and the vLLM API guard has unit
    tests. Real vLLM/CUDA combinations still require CUDA hardware coverage.
22. **Resolved in this pass:** coverage, mypy, and vulture paths now target the
    actual `bench` tree; optional benchmark/research paths are explicitly
    separated from the core gate.
23. **Resolved in this pass:** CI now includes a required frontend job covering
    lint, TypeScript checking, production dependency auditing, the production
    build, and Chromium smoke tests. The Pages deployment workflow runs the
    same production audit and smoke tests before publishing. Python
    Ruff/mypy/test checks remain authoritative in the matrix job.
24. **Resolved for the current local/frontend toolchain:** CI audits npm
    dependencies and locked Python dependencies with `pip-audit`, and uploads
    CycloneDX SBOMs for Python and npm. The release workflow requests GitHub
    build provenance attestations. Vite 8 and Tailwind 4 are now installed
    (Node.js >=22.12 is declared) and
    the full npm audit is clean; the remaining supply-chain work is to execute
    the same locked checks in GitHub and retain the artifacts.

### P1 — API and architecture

25. **Mostly resolved in this pass:** adapters now expose a polymorphic
    attach/detach context-manager contract with immutable lifecycle state, and
    the cache backend contract is explicit. Metadata now carries a versioned
    serialization schema and strict imports reject incompatible files; device
    ownership still needs a formal value object.
26. **Mostly resolved in this pass:** immutable method descriptors now bind
    method, config, implementation, and capabilities in the dispatcher, and
    `kvfold.method_descriptors()` exposes that canonical registry publicly.
    Site and long-form documentation still need generated consumption rather
    than hand-maintained copy.
27. **Partially resolved in this pass:** `CacheBackend` is an injectable storage
    port and adapter lifecycle is separated from the API handle. Allocator and
    observability ports remain future work.
28. **Mostly resolved in this pass:** adapter lifecycle, cache storage,
    allocator strategies (`Allocator`), and quantizers (`Quantizer` protocol)
    now have explicit polymorphic interfaces. Device ownership and a unified
    observability port remain to be extracted.
29. **Partially resolved in this pass:** lifecycle transitions are explicit,
    adapters expose immutable state snapshots, and compression metrics provide
    locked atomic recording/snapshot methods. The public `CompressionHandle` is
    now frozen against reference replacement; its adapter remains the
    deliberately mutable lifecycle owner.
30. **Resolved for the canonical API in this pass:** public compatibility
    aliases for compressor names, metadata ratios, cache implementation names,
    adapter exports, and model-family shims were removed. The style guide now
    requires explicit API breaks and underscored implementation state.
31. **Mostly resolved in this pass:** current extension/style/architecture
    docs now describe the canonical package, ports, lifecycle, and naming policy;
    stale current method/module references were corrected, and a relative-link
    checker now runs in CI. Constructor/module exceptions are documented;
    generated API docs remain.

### P2 — docs and product communication

32. **Partially resolved in this pass:** current CLI/docs/examples now use the
    canonical `kvfold` package and registered method names (`flash`, `low`,
    `pass`); migration and research pages intentionally retain historical names.
    A generated documentation/link check and final legacy-reference review are
    still required.
33. **Mostly resolved in this pass:** site and current user-performance copy now
    scopes quality language to reported paper regimes and links benchmark
    reproduction commands. Historical research/migration text intentionally
    preserves the paper's terminology but is not a deployment guarantee.
34. **Mostly resolved in this pass:** benchmark JSON artifacts record
    platform/software/device/seed/warmup/iteration metadata, speed results
    include standard deviation, 95% confidence-interval estimates, and extrema,
    and memory results include logical plus resident tensor-storage bytes.
    CUDA allocator fragmentation and process-level reserved memory remain
    deployment-specific follow-up work.
35. **Mostly resolved in this pass:** CI and the new tag/manual release
    workflow build, validate, upload, and install-test wheels, emit provenance
    attestations, and provide a protected, manual PyPI trusted-publishing job.
    The repository owner still has to configure the matching PyPI publisher and
    GitHub `pypi` environment before that job can succeed.
36. **Resolved in this pass:** `SECURITY.md` now defines supported versions,
    private-reporting expectations, dependency scanning, and the boundary around
    model data, cache tensors, generated text, and telemetry.

### P2 — live site UX/UI

37. **Resolved in the local source in this pass:** responsive navigation now
    provides an accessible mobile menu.
38. **Resolved in the local source in this pass:** the fold mark, wordmark,
    favicon, and OG image use one visual system.
39. **Resolved in the local source in this pass:** hierarchy, contrast, spacing,
    and the technical narrative were redesigned; visual review on all target
    devices remains a release check.
40. **Resolved in the local source in this pass:** Docs points to repository
    documentation and benchmark evidence links to the protocol.
41. **Resolved in the local source in this pass:** FAQ buttons and answers have
    linked IDs and ARIA state.
42. **Resolved in the local source in this pass:** benchmark cards now include a
    semantic table of values and reproduction commands.
43. **Resolved in this pass:** the theme applies before hydration, follows later
    OS preference changes when no explicit user choice exists, and preserves an
    explicit stored choice.
44. **Resolved for the critical paths in this pass:** frontend lint, typecheck,
    production build, and Chromium smoke tests cover navigation, theme toggle,
    FAQ semantics, responsive menu, external evidence links, and reduced motion.

## Remediation plan

### Phase 0 — make truth observable

1. Establish a reproducible developer environment (`uv.lock` or pinned
   requirements), make `scripts/setup.sh` idempotent without destructive default
   behaviour, and publish `python -m ...` commands that work from a clean clone.
2. Fix stale package paths and consolidate CI into required lint, type, unit,
   integration/nightly, GPU, site, packaging, and security jobs.
3. Choose and enforce one coverage threshold; generate an HTML/XML report and
   publish test/benchmark provenance. The fast unit gate is 90% over the core
   package; optional vLLM/Triton and benchmark/research entry points require
   their own integration/GPU coverage.
4. Add a compatibility matrix and release smoke test from a built wheel.

### Phase 1 — harden the runtime

1. Extract a `CacheBackend`/`CompressorStrategy`/`AllocatorStrategy` contract and
   immutable configuration/value objects.
2. Use the supported model-scoped framework hook, lock lifecycle transitions,
   and make restoration exact. Do not reintroduce process-wide monkey patches.
3. Add transactional cache operations, invariant validation, structured errors,
   and logical/resident memory accounting.
4. Add CPU edge-case, concurrency, import-order, multi-model, CUDA, Triton, and
   vLLM compatibility tests.

### Phase 2 — documentation and API quality

1. Generate API docs from the canonical `kvfold` package and repair all links and
   examples.
2. Centralize method descriptors and expose supported capabilities programmatically.
3. Apply Google-style docstrings, `_internal` naming, type checking, and
   deprecation policy consistently.
4. Rework benchmark reporting so claims are scoped, reproducible, and versioned.

### Phase 3 — product surface

1. Ship the new responsive navigation and unified logo/mark assets.
2. Replace absolute performance language with evidence-led copy, add accessible
   benchmark tables, real documentation links, and copyable install commands.
3. Add frontend lint and browser smoke tests, then validate at mobile, tablet,
   desktop, light/dark, keyboard, reduced-motion, and screen-reader-oriented
   states.

### Definition of done

The project is ready to call production-ready only when a clean clone can install
the pinned environment, all required checks pass on the declared support matrix,
the adapter lifecycle is isolated and reversible, GPU/vLLM claims have matching
verification, benchmark claims have provenance, and the site communicates those
limits accurately on all supported viewport and accessibility states.
