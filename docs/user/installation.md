# Installation

## Requirements

- Python 3.11 or newer
- PyTorch 2.5 or newer
- transformers 4.45 or newer

## pip

```bash
pip install kvfold
```

## Optional extras

```bash
pip install "kvfold[triton]"   # Triton kernels for reconstruction / JL
pip install "kvfold[vllm]"     # vLLM adapter (Shape A: export/import helpers; Shape B: KVCacheOffloadWorker subclass requires a CUDA box to validate)
pip install "kvfold[bench]"    # matplotlib, datasets, pandas for benchmarks
pip install "kvfold[dev]"      # pytest, ruff, mypy, hypothesis
pip install "kvfold[docs]"     # mkdocs for documentation
```

## From source

```bash
git clone https://github.com/sachncs/kvcompress
cd kvcompress
pip install -e ".[dev,bench,docs]"
```

## Verifying the install

```bash
kvfold version
kvfold validate
```

`validate` runs a smoke test on synthetic K/V and (if `transformers` is
installed) on a small HF model.
