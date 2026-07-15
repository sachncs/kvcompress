# Installation

## Requirements

- Python 3.12 or newer
- PyTorch 2.5 or newer
- transformers 4.45 or newer

## pip

```bash
pip install kvcompress
```

## Optional extras

```bash
pip install "kvcompress[triton]"   # Triton kernels for reconstruction / JL
pip install "kvcompress[vllm]"     # vLLM adapter (Shape A: export/import helpers; Shape B: KVCacheOffloadWorker subclass requires a CUDA box to validate)
pip install "kvcompress[bench]"    # matplotlib, datasets, pandas for benchmarks
pip install "kvcompress[dev]"      # pytest, ruff, mypy, hypothesis
pip install "kvcompress[docs]"     # mkdocs for documentation
```

## From source

```bash
git clone https://github.com/anomalyco/kvcompress
cd kvcompress
pip install -e ".[dev,bench,docs]"
```

## Verifying the install

```bash
kvcompress version
kvcompress validate
```

`validate` runs a smoke test on synthetic K/V and (if `transformers` is
installed) on a small HF model.