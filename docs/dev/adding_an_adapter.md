# Adding an adapter

Most model families don't need any custom code — the `DynamicCache`
interception in `HuggingFaceAdapter` covers them. You only need a shim
when:

- The model uses a non-standard cache layout (e.g., MLA on DeepSeek).
- The model needs an explicit RoPE / norm hook for correct key
  compression.
- The model's `past_key_values` type isn't `DynamicCache`.

For all other cases, the shim is just a documented no-op so the registry
has a place to dispatch to.

## 1. Write the shim

```python
# kvcompress/adapters/my_family.py
from __future__ import annotations

from typing import Any


def install(model: Any, cache_manager: Any) -> None:
    """Install JoLT compression for MyFamily models.

    MyFamily uses standard GQA with rotary embeddings; the DynamicCache
    subclass installed by HuggingFaceAdapter already handles the cache
    correctly. This shim is a no-op.
    """
    return None
```

If the family *does* need a custom hook, replace `return None` with the
real logic. For example, to force pre-RoPE key compression on a family
that post-RoPEs by default:

```python
def install(model, cache_manager):
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "rope_mode"):
            layer.self_attn.rope_mode = "pre"
    return None
```

## 2. Register

```python
# kvcompress/adapters/registry.py
_REGISTRY: dict[str, str] = {
    ...
    "my-family": "kvcompress.adapters.my_family",
}
```

## 3. Tests

```python
# tests/unit/adapter_test.py
def test_my_family_resolves():
    from kvcompress.adapters.registry import resolve
    assert resolve("my-family") == "kvcompress.adapters.my_family"


def test_my_family_install_noop():
    from kvcompress.adapters.my_family import install
    # Should not raise even though model/cache_manager are arbitrary objects.
    install(model=None, cache_manager=None)
```

That's it. The auto-detection in `HuggingFaceAdapter.enable()` reads
`model.config.model_type` and dispatches to the right shim.