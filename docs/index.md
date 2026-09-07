# kvcompress documentation

> **Authorship disclaimer.** sachin is the **implementer** of this
> library, not an author of the JoLT paper. The algorithm is described
> in
> *Krishnan, R. & Schulz, V. (2026). "A JoLT for the KV Cache: Near-Lossless
> KV Cache Compression via Joint Tucker and JL-Residual Allocation for
> LLMs." arXiv:2607.12550.*
>
> This implementation is a **third-party re-implementation** by sachin.
> The code follows the paper's algorithm but makes no claim to the
> underlying theoretical results. If you publish work that uses this
> library, please cite the original paper, not this repository.

`kvcompress` implements the JoLT algorithm (paper:
[arXiv:2607.12550](https://arxiv.org/abs/2607.12550)) plus its
randomized-SVD fast variant FlashJoLT, and provides a generic
KV-cache-compression interface for any decoder-only LLM.

## Quick links

| I want to… | Read this |
|---|---|
| Compress my model's KV cache | [quickstart.md](quickstart.md) |
| Understand the algorithm | [research/math.md](research/math.md) + [research/algorithm.md](research/algorithm.md) |
| Pick a compression ratio | [user/performance_guide.md](user/performance_guide.md) |
| See the API surface | [user/api.md](user/api.md) |
| Add a new compressor | [dev/adding_a_compressor.md](dev/adding_a_compressor.md) |
| Add a new model family | [dev/adding_an_adapter.md](dev/adding_an_adapter.md) |
| Run the benchmarks | [benchmarks/running.md](benchmarks/running.md) |
| Reproduce the paper's numbers | [research/reproduction_notes.md](research/reproduction_notes.md) |
| Integrate with vLLM | [research/paper_notes.md](research/paper_notes.md#vllm-integration) |

## Sections

### [User guide](user/index.md)

Install, quickstart, API reference, compression methods, performance guide,
long context, troubleshooting, FAQ. The "I just want to use it" path.

### [Developer guide](dev/index.md)

Setup, architecture, style, testing, adding a compressor, adding an
adapter, release, debugging. The "I want to extend it" path.

### [Researcher notes](research/index.md)

Math restating Eqs. 1-4 of the paper, algorithm walkthrough, spectral
motivation, free-zone analysis, ablations, comparison with baselines,
reproduction notes, paper notes. The "I want to understand it" path.

### [Benchmarks](benchmarks/overview.md)

Memory, throughput, reconstruction fidelity, long context (needle +
RULER). The "I want to measure it" path.

## Library overview

`kvcompress` is built around three objects:

1. **`KVCompressor`** — the algorithm. Implements `compress(K, V) ->
   (CompressedPayload, CompressedPayload)` and the inverse. Concrete
   classes: `JoLTCompressor`, `FlashJoLTCompressor` (default),
   `IdentityCompressor`, `LowRankCompressor`, `IntQuantOnlyCompressor`.

2. **`CompressedKVCache` / `CacheManager`** — the storage. Holds the
   compressed payloads per layer; LRU eviction; lazy reconstruction
   on read. The cache is the only stateful piece.

3. **`HuggingFaceAdapter`** — the integration. Patches `DynamicCache` so
   any HF causal LM gets compression transparently during `generate()`.
   vLLM has its own integration (`adapters/vllm.py` and
   `adapters/vllm_kv_offload.py`).

The contract between them is small:

```python
class KVCompressor(ABC):
    def compress(self, K, V) -> tuple[CompressedPayload, CompressedPayload]: ...
    def decompress(self, kp, vp) -> tuple[Tensor, Tensor]: ...
    def estimate_size(self, payload) -> int: ...
    def stats(self) -> dict: ...
```

That's the entire surface. New compressors plug in by implementing these
four methods. See [dev/adding_a_compressor.md](dev/adding_a_compressor.md).

## Algorithm summary

The JoLT algorithm is partial Tucker decomposition with a JL-rotated
residual. Concretely, for each layer's K (and V) tensor of shape
`(m, T, dh)`:

1. **Pin** the head/layer mode (`m` axis) — ST-HOSVD truncates only the
   token (`T`) and feature (`dh`) modes. The paper shows pinning
   doesn't lose accuracy (Appendix B.2) and is 2.4-3.0× faster than
   full HOOI.

2. **Truncate** to ranks `(r_T, r_d)` chosen by the joint allocator.
   Cost in bytes: `(m·r_T·r_d + T·r_T + dh·r_d)·c` (Eq. 1).

3. **JL-rotate the residual** `R = X - X̂`, then **uniformly quantize**
   at `b` bits. The JL rotation makes the residual's energy uniform
   across dimensions, so uniform `b`-bit quantisation gives roughly
   uniform `b`-bit error.

4. **Allocate** `(r_T, r_d, b)` per-(K, V) cell by Lagrangian
   bisection. Each cell's grid `(r_T, r_d, b)` is enumerated
   independently; `λ` is found by bisection to hit the global byte
   budget. The result: near-lossless 2-3× compression in the
   paper's "free zone" without per-cell hyperparameter tuning.

See [research/math.md](research/math.md) for the equations,
[research/algorithm.md](research/algorithm.md) for the code walkthrough,
and [research/free_zone.md](research/free_zone.md) for when it works.

## Quality at HEAD

| Metric | Value |
|---|---|
| Tests | 237 pass, 4 skipped |
| Line coverage | 73% |
| `black --check` | clean |
| `ruff check` | clean |
| `ruff format --check` | clean |
| `mypy src` | clean |
| `vulture src/` | 2 hits at 100% (intentional `if False: # TYPE_CHECKING` patterns) |

## What's not in scope

- **Encoder-decoder models.** Works on the cache level but the allocator
  assumes a single `(m, T, dh)` layout per cell. Cross-attention KV
  caches don't fit that pattern.
- **Training.** JoLT is inference-only. The cache writer doesn't
  track gradients; don't enable compression during a training loop.
- **Multi-modal / state-space.** The cache layout assumes a standard
  transformer attention cache. Mamba and similar don't have one.
- **Custom CUDA kernels.** The library ships a real Triton kernel
  for the fused Tucker reconstruction. A hand-written CUDA kernel
  could buy 2-3× more on NVIDIA, but isn't shipped.

See [research/paper_notes.md](research/paper_notes.md) for the full
list of caveats.