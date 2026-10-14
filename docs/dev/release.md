# Release process

## Versioning

We follow [Semantic Versioning](https://semver.org/):

- **Major** (X.0.0): breaking API changes.
- **Minor** (0.X.0): new features, backward-compatible.
- **Patch** (0.0.X): bug fixes.

The current version is in `kvfold/__init__.py`:

```python
__version__ = "0.2.0"
```

## Pre-release checklist

1. `ruff check kvfold tests examples scripts` — no warnings.
2. `ruff format --check kvfold tests examples scripts` — formatted.
3. `mypy kvfold` — no errors.
4. `pytest -m "not slow and not integration and not gpu"` — all pass.
5. `uv run pytest -m integration` — all network integration tests pass.
6. `uv run pytest -m gpu` — required before making CUDA/Triton claims, on a
   documented GPU environment.
7. `uv run pytest --cov=kvfold --cov-report=term-missing` — coverage ≥ 90%.
8. `kvfold validate` — runs cleanly.
9. `scripts/reproduce_paper_numbers.sh` — JSON outputs land in `results/`.

## Commit & tag

```bash
git checkout master
git pull
# Bump version in __init__.py
git commit -am "release: 0.X.0"
git tag v0.X.0
git push origin master --tags
```

## Publish to PyPI

The GitHub release workflow uses PyPI trusted publishing. Configure the PyPI
project's pending publisher for this repository and the protected `pypi`
environment before enabling publication. Publishing is deliberately manual from
the workflow dispatch input; building a tag does not publish automatically.

For normal releases, use the protected GitHub Actions workflow: it builds and
checks the artifacts, then only publishes when the `publish` workflow input is
explicitly enabled. This keeps PyPI credentials out of developer machines.

For an emergency local publication, use a short-lived PyPI token and verify the
artifact before uploading:

```bash
uv build
uvx twine check dist/*
uvx twine upload dist/*
```

## Post-release

1. GitHub release with the changelog excerpt.
2. Verify `pip install kvfold` works for `0.X.0`.
3. Announce on relevant channels.
