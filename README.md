# kvcompress

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-2607.12550-red)](https://arxiv.org/abs/2607.12550)

Universal plug-and-play **KV cache compression** for decoder-only LLMs.

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

`kvcompress` implements the **JoLT** algorithm from
[*A JoLT for the KV Cache* (arXiv:2607.12550)](https://arxiv.org/abs/2607.12550)
plus its randomized-SVD fast variant **FlashJoLT**. The library is built around a
generic `KVCompressor` interface so future compressors (quantization,
sparsity, low-rank) plug in without touching the cache or adapter code.

## Why JoLT

The KV cache is the dominant memory cost of transformer inference at long
context. JoLT treats each layer's cache as a third-order tensor
`(heads, tokens, features)` and exploits a robust asymmetry the paper measures
empirically:

- **Head and layer axes are essentially incompressible.** Don't spend budget.
- **Token and feature axes carry almost all the redundancy.** Compress them.
- **Values are 2-3× harder to compress than keys.** Allocate ranks and bits
  separately per K/V.

The result is a near-lossless 2-3× free zone that holds across perplexity,
GSM8K, and RULER needle retrieval on both GQA (Mistral-7B) and MHA
(LLaMA-2-13B). FlashJoLT cuts compression time 5-13× at matched quality.

## Install

```bash
pip install kvcompress

# Optional extras
pip install "kvcompress[triton]"    # Triton kernels for reconstruction / JL
pip install "kvcompress[vllm]"      # vLLM adapter
pip install "kvcompress[bench]"     # matplotlib, pandas, datasets
pip install "kvcompress[dev]"       # pytest, ruff, mypy, hypothesis
pip install "kvcompress[docs]"      # mkdocs
```

## Quickstart

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
enable_compression(model, method="jolt",       compression_ratio=3.0)
enable_compression(model, method="flashjolt",  compression_ratio=2.5)
enable_compression(model, method="lowrank",    rank=128)
enable_compression(model, method="int4",       per_channel=True)
enable_compression(model, method="fp8")
enable_compression(model, method="identity")   # baseline passthrough
```

## CLI

```bash
kvcompress benchmark --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --method flashjolt
kvcompress profile   --model Qwen/Qwen2.5-0.5B-Instruct
kvcompress compress  --model ... --input prompt.txt --output kv.pt
kvcompress validate  --model ...
```

## Architecture

```
KVCompressor (ABC)
   ├── JoLTCompressor          # partial Tucker + JL residual + Lagrangian
   ├── FlashJoLTCompressor     # randomized SVD token mode
   ├── LowRankCompressor       # matrix SVD baseline
   ├── IntQuantCompressor      # INT2/4/8
   ├── FP8QuantCompressor
   └── IdentityCompressor

CompressedKVCache  ── layer-indexed, lazy decompress, eviction ──
JointAllocator      ── per-(group, K/V) Lagrangian, bisection on λ
HF Adapter          ── DynamicCache interception, family shims
```

## Tested showcase models

The library has been validated end-to-end on these small models:

| Family | Model | Notes |
|---|---|---|
| Llama | TinyLlama-1.1B | MHA, single-GPU friendly |
| Qwen | Qwen2.5-0.5B-Instruct | GQA, RoPE |
| GPT-2 | gpt2 | Sanity baseline |

Paper numbers (Mistral-7B, LLaMA-2-13B) require a GPU box with ≥40 GB and are
documented in `docs/research/reproduction_notes.md`. The algorithm itself is
validated on synthetic tensors (see `scripts/run_table2_reconstruction.py`).

## Documentation

- [Quickstart](docs/quickstart.md)
- [API reference](docs/user/api.md)
- [Algorithm and math](docs/research/math.md)
- [Architecture](docs/dev/architecture.md)
- [Developer guide](docs/dev/setup.md)
- [Researcher notes](docs/research/paper_notes.md)
- [Benchmarks](docs/benchmarks/overview.md)

## Citation

```bibtex
@article{krishnan2026jolt,
  title={A JoLT for the KV Cache: Near-Lossless KV Cache Compression via
         Joint Tucker and JL-Residual Allocation for LLMs},
  author={Krishnan, Rahul and Schulz, Volker},
  journal={arXiv preprint arXiv:2607.12550},
  year={2026}
}
```

## License

Apache-2.0.