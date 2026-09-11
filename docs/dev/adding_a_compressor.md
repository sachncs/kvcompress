# Adding a compressor

This walkthrough adds a new compressor to `kvfold`. The pattern is
the same whether you're adding a research variant of JoLT, a baseline
quantizer, or a hybrid method.

## 1. Subclass `Compressor`

```python
# kvfold/core/my_method.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from kvfold.core.base import Compressor, Payload, Stats


@dataclass
class MyMethodCompressor(Compressor):
    """One-line description of the method.

    Optional knobs:
        arg_a: int — what it does.
        arg_b: bool — what it does.
    """

    method = "my-method"
    arg_a: int = 16
    arg_b: bool = True

    def __post_init__(self) -> None:
        # Validate args and initialise any sub-components.
        if self.arg_a <= 0:
            raise ValueError(f"arg_a must be > 0, got {self.arg_a}")

    def compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[Payload, Payload]:
        """Compress a (key, value) pair into two payloads."""
        from kvfold.core.base import Compressor
        Compressor.validate(key, value)

        k_payload = self._compress_one(key)
        v_payload = self._compress_one(value)
        return k_payload, v_payload

    def restore(
        self,
        key_payload: Payload,
        value_payload: Payload,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._decompress_one(key_payload), self._decompress_one(value_payload)

    def _compress_one(self, x: torch.Tensor) -> Payload:
        # ... your algorithm here ...
        factors = some_low_rank_approximation(x, rank=self.arg_a)
        return Payload(
            method=self.method,
            shape=tuple(x.shape),
            dtype=x.dtype,
            metadata={"r": self.arg_a},
            data={"factors": factors},
            stats=Stats(
                bytes_original=x.numel() * x.element_size(),
                bytes_compressed=factors.numel() * factors.element_size(),
            ),
        )

    def _decompress_one(self, payload: Payload) -> torch.Tensor:
        factors = payload.data["factors"]
        # ... inverse of your algorithm ...
        return reconstruction.to(payload.dtype)
```

## 2. Wire into the dispatch

```python
# kvfold/core/builtins.py
from kvfold.core.dispatch import REGISTRY
from kvfold.core.my_method import MyMethodCompressor

REGISTRY.register("my-method", MyMethodCompressor)
```

## 3. Export from the package

```python
# kvfold/__init__.py
LAZY_EXPORTS = {
    ...
    "MyMethodCompressor": ("kvfold.core.my_method", "MyMethodCompressor"),
}
```

## 4. Tests

```python
# tests/unit/my_method_test.py
def test_roundtrip():
    K = torch.randn(4, 16, 8)
    V = torch.randn(4, 16, 8)
    comp = MyMethodCompressor()
    kp, vp = comp.compress(K, V)
    k_hat, v_hat = comp.restore(kp, vp)
    assert k_hat.shape == K.shape


def test_bytes_reduced():
    K = torch.randn(4, 16, 8)
    V = torch.randn(4, 16, 8)
    comp = MyMethodCompressor(arg_a=2)
    kp, vp = comp.compress(K, V)
    original = K.numel() * K.element_size() * 2
    assert kp.bytes_compressed + vp.bytes_compressed < original


def test_in_registry():
    from kvfold import build_compressor
    c = build_compressor("my-method")
    assert isinstance(c, MyMethodCompressor)
```

## 5. Document

Add a row to the table in `docs/user/compression_methods.md` and write
a section in `docs/research/math.md` if the method has novel theory.

That's it. The HF adapter picks up your method automatically because it
goes through the `Compressor` ABC; you don't need to touch the cache
or the adapter code.
