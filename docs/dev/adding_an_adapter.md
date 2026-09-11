# Adding an adapter

Most model families don't need any custom code — the `DynamicCache`
interception in `HF` covers them. You only need a shim when:

- The model uses a non-standard cache layout (e.g., MLA on DeepSeek).
- The model needs an explicit RoPE / norm hook for correct key
  compression.
- The model's `past_key_values` type isn't `DynamicCache`.

For all other cases, the shim is just a documented no-op so the registry
has a place to dispatch to.

## 1. Write the shim

```python
# kvfold/adapter/my_family.py
from __future__ import annotations

from typing import Any


def install(model: Any, pool: Any) -> None:
    """Install compression for MyFamily models.

    MyFamily uses standard GQA with rotary embeddings; the DynamicCache
    subclass installed by HF already handles the cache correctly.
    This shim is a no-op.
    """
    return None
```

If the family *does* need a custom hook, replace `return None` with the
real logic. For example, to force pre-RoPE key compression on a family
that post-RoPEs by default:

```python
def install(model, pool):
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "rope_mode"):
            layer.self_attn.rope_mode = "pre"
    return None
```

## 2. Register

```python
# kvfold/adapter/registry.py
from kvfold.adapter.registry import Family, register


@register
class MyFamily(Family):
    name = "my-family"

    def install(self, model, pool):
        from kvfold.adapter.my_family import install as _install
        return _install(model, pool)
```

The registry is populated at import time by `_register_builtins()`; the
new family is added by appending an entry to the `_register_builtins`
loop in `kvfold/adapter/registry.py`.

## 3. Tests

```python
# tests/unit/adapter_test.py
def test_my_family_resolves():
    from kvfold.adapter.registry import known_model_types, resolve
    assert "my-family" in known_model_types()
    assert resolve("my-family") is not None


def test_my_family_install_noop():
    from kvfold.adapter.my_family import install
    # Should not raise even though model/pool are arbitrary objects.
    install(model=None, pool=None)
```

That's it. The auto-detection in `HF.enable()` reads
`model.config.model_type` and dispatches to the right shim.
