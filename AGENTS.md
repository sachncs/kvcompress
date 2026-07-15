# Agent instructions for kvcompress

## Build & lint commands

- Lint: `ruff check src tests examples scripts`
- Format: `ruff format src tests examples scripts`
- Type check: `mypy src`
- Run tests (unit, fast): `pytest -m "not slow and not integration and not gpu"`
- Run all tests: `pytest`
- Coverage: `pytest --cov=kvcompress --cov-report=term-missing`

## Code conventions

- Python 3.12+, full type hints, dataclasses preferred over tuples.
- One module-level logger per file: `log = logging.getLogger(__name__)`.
- No global mutable state. Pass dependencies explicitly.
- Public APIs documented with docstrings (Google-style).
- No `print()` in library code; use `log.info` / `log.debug`.
- Avoid `from X import *`. Prefer `from X import name`.
- Tensor shapes documented in docstrings when non-obvious.
- Never hardcode batch size, head count, or feature dim.

## Reference format

When citing code locations in conversation, use `path/to/file.py:42` form so
the user can navigate directly.

## PyTorch conventions

- All algorithm code is pure PyTorch, no hidden device transfer.
- Use `torch.Generator` for randomness, never `torch.manual_seed` as a side
  effect inside library functions.
- Quantize / dequantize must be reversible up to numerical noise.

## Tests

- Unit tests live in `tests/unit/<module>_test.py`.
- Integration tests in `tests/integration/` (marked `@pytest.mark.integration`).
- Property tests use Hypothesis; aim for ≥ 95 % coverage on `src/kvcompress/`.
- Fixtures in `tests/fixtures/` are tiny models, not real LLMs.

## Commits

One atomic commit per logical impact. A milestone may have many commits.
Commit messages: short imperative summary, then a body if needed.

## Don'ts

- Don't commit model weights, large fixtures, or generated plots.
- Don't push to remote without explicit instruction.
- Don't modify global git config.
- Don't run large model downloads without explicit instruction.
