# Style guide

## Python style

- **Version:** Python 3.12+. No compatibility shims for older versions.
- **Type hints:** full annotations on every public symbol. Internal
  helpers may use less strict types.
- **Dataclasses:** preferred over `NamedTuple` or plain dicts for any
  record with > 2 fields.
- **Imports:** `from X import name` over `from X import *`. No relative
  imports across packages.
- **Logging:** one module-level logger per file:
  `log = logging.getLogger(__name__)`. Use `log.info`/`log.debug`,
  never `print()` in library code.

## Linting and formatting

We use [`ruff`](https://docs.astral.sh/ruff/) for both lint and format:

```bash
ruff check src tests examples scripts
ruff format src tests examples scripts
```

Configuration is in `pyproject.toml` and `ruff.toml`. The CI gate fails
on any `ruff check` warning.

## Type checking

We use `mypy` in non-strict mode (the surface area is large; strict
mode is too brittle for our pace). Public symbols should still be fully
annotated. CI runs:

```bash
mypy src
```

## Docstrings

Google-style docstrings on every public class, method, and function.
Module-level docstrings describe the file's purpose in one sentence plus
optional longer description.

```python
def compress(
    self,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[CompressedPayload, CompressedPayload]:
    """Compress a (key, value) pair into two payloads.

    Args:
        key: tensor of shape ``(m, T, dh)`` where ``m`` merges head and
            layer, ``T`` is the token axis, ``dh`` is per-head feature dim.
        value: same shape as ``key``.

    Returns:
        Two payloads: (key payload, value payload).

    Raises:
        ValueError: if K and V have different shapes or non-3-D layouts.
    """
```

## Tensor shapes

Always document the shape with the einsum letters the algorithm uses
(see `compressor/tucker.py`). The convention is:

- `m` — merged head × layer count.
- `T` — token axis length.
- `dh` — per-head feature dim.
- `rT` — token-mode rank.
- `rd` — feature-mode rank.

## No global state

Pass dependencies explicitly. The only allowed exception is the JL
projection cache (`compressor.jl._PROJECTION_CACHE`) which is keyed by
shape+seed and shared across the process for performance.

## Commit messages

Short imperative summary, then an optional body. Examples:

```
M5: JoLT compressor + cache round-trip

- compressor/jolt.py: ties ST-HOSVD, JL residual, allocator together
- cache/compress.py + cache/manager.py + cache/metadata.py
- 14 tests including round-trip, byte reduction, identity at bits=0
```

Atomic commits — one logical impact per commit. A milestone may span
many commits.