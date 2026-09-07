# Release process

## Versioning

We follow [Semantic Versioning](https://semver.org/):

- **Major** (X.0.0): breaking API changes.
- **Minor** (0.X.0): new features, backward-compatible.
- **Patch** (0.0.X): bug fixes.

The current version is in `kvcompress/__init__.py`:

```python
__version__ = "0.1.0"
```

## Pre-release checklist

1. `ruff check src tests examples scripts` — no warnings.
2. `ruff format --check src tests examples scripts` — formatted.
3. `mypy src` — no errors.
4. `pytest -m "not slow and not integration and not gpu"` — all pass.
5. `pytest` (including integration) — all pass on a GPU box.
6. `pytest --cov=kvcompress --cov-report=term-missing` — coverage ≥ 80%.
7. `kvcompress validate` — runs cleanly.
8. `scripts/reproduce_paper_numbers.sh` — JSON outputs land in `results/`.

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

```bash
pip install build twine
python -m build
twine upload dist/*
```

## Post-release

1. GitHub release with the changelog excerpt.
2. Verify `pip install kvcompress` works for `0.X.0`.
3. Announce on relevant channels.