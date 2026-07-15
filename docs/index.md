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

The documentation is split into four sections:

* [User guide](user/index.md) — installation, quickstart, API reference.
* [Developer guide](dev/index.md) — architecture, adding a compressor,
  testing, releasing.
* [Researcher notes](research/index.md) — algorithm, math, free-zone
  analysis, reproduction notes.
* [Benchmarks](benchmarks/overview.md) — running benchmarks, interpreting
  results.

If you're new to `kvcompress`, start with the [quickstart](quickstart.md).