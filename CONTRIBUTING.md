# Contributing

Thanks for your interest in contributing to kvcompress.

## Quick links

- [Setup](docs/dev/setup.md)
- [Architecture](docs/dev/architecture.md)
- [Style](docs/dev/style.md)
- [Testing](docs/dev/testing.md)
- [Adding a compressor](docs/dev/adding_a_compressor.md)
- [Adding an adapter](docs/dev/adding_an_adapter.md)

## Development workflow

1. Fork and create a topic branch.
2. Make atomic commits with imperative-mood messages.
3. Run `ruff check src tests examples scripts` and `ruff format` before committing.
4. Run `mypy src` and `pytest -m "not slow and not integration and not gpu"`.
5. Open a pull request referencing any related issue.

## Coding principles

- The compressor algorithm and the cache storage are decoupled. New
  compressors implement the `KVCompressor` ABC and need nothing else.
- Adapter code is the *only* code allowed to import from `transformers`.
- No global mutable state. Anything stochastic takes a seed.
- All public APIs are documented with Google-style docstrings.

## Reporting bugs

Please include:

- Minimal reproduction script.
- `kvcompress --version` output.
- Python, PyTorch, transformers versions.
- Hardware (CPU / GPU / MPS).