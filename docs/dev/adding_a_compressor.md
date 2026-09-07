# Adding a compressor

This walkthrough adds a new compressor to `kvcompress`. The pattern is
the same whether you're adding a research variant of JoLT, a baseline
quantizer, or a hybrid method.

## 1. Subclass `KVCompressor`

```python
# kvcompress/compressor/my_method.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from kvcompress.compressor.base import (
    CompressedPayload,
    CompressorStats,
    KVCompressor,
)


@dataclass
class MyMethodCompressor(KVCompressor):
    """One-line description of the method.

    Optional knobs:
        arg_a: int — what it does.
        arg_b: bool — what it does.
    """

    name = "my-method"
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
    ) -> tuple[CompressedPayload, CompressedPayload]:
        """Compress a (key, value) pair into two payloads."""
        if key.shape != value.shape:
            raise ValueError(f"K/V shape mismatch: {key.shape} vs {value.shape}")
        if key.dim() != 3:
            raise ValueError(
                f"MyMethod expects 3-D (m, T, dh); got {tuple(key.shape)}"
            )

        k_payload = self._compress_one(key)
        v_payload = self._compress_one(value)
        return k_payload, v_payload

    def decompress(
        self,
        key_payload: CompressedPayload,
        value_payload: CompressedPayload,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._decompress_one(key_payload), self._decompress_one(value_payload)

    def _compress_one(self, x: torch.Tensor) -> CompressedPayload:
        # ... your algorithm here ...
        factors = some_low_rank_approximation(x, rank=self.arg_a)
        return CompressedPayload(
            method=self.name,
            shape=tuple(x.shape),
            dtype=x.dtype,
            metadata={"r": self.arg_a},
            data={"factors": factors},
            stats=CompressorStats(
                bytes_original=x.numel() * x.element_size(),
                bytes_compressed=factors.numel() * factors.element_size(),
            ),
        )

    def _decompress_one(self, payload: CompressedPayload) -> torch.Tensor:
        factors = payload.data["factors"]
        # ... inverse of your algorithm ...
        return reconstruction.to(payload.dtype)
```

## 2. Wire into the dispatch

```python
# kvcompress/adapters/huggingface.py
def _build_compressor(method: str, **kwargs: Any) -> KVCompressor:
    method = method.lower()
    if method == "my-method":
        from kvcompress.compressor.my_method import MyMethodCompressor
        return MyMethodCompressor(**kwargs)
    ...
```

## 3. Export from the package

```python
# kvcompress/__init__.py
_LAZY_EXPORTS = {
    ...
    "MyMethodCompressor": ("kvcompress.compressor.my_method", "MyMethodCompressor"),
}
```

## 4. Add a CLI alias (optional)

```python
# kvcompress/api.py
MethodName = Literal[
    "jolt", "flashjolt", "lowrank",
    "int2", "int4", "int8",
    "fp8", "fp16", "bf16",
    "identity",
    "my-method",  # <-- here
]
```

## 5. Tests

```python
# tests/unit/my_method_test.py
def test_roundtrip():
    K = torch.randn(4, 16, 8)
    V = torch.randn(4, 16, 8)
    comp = MyMethodCompressor()
    kp, vp = comp.compress(K, V)
    k_hat, v_hat = comp.decompress(kp, vp)
    assert k_hat.shape == K.shape


def test_bytes_reduced():
    K = torch.randn(4, 16, 8)
    V = torch.randn(4, 16, 8)
    comp = MyMethodCompressor(arg_a=2)
    kp, vp = comp.compress(K, V)
    original = K.numel() * K.element_size() * 2
    assert kp.bytes_compressed + vp.bytes_compressed < original


def test_in_registry():
    from kvcompress.api import _build_compressor
    c = _build_compressor("my-method")
    assert isinstance(c, MyMethodCompressor)
```

## 6. Document

Add a row to the table in `docs/user/compression_methods.md` and write
a section in `docs/research/math.md` if the method has novel theory.

That's it. The HF adapter picks up your method automatically because it
goes through the `KVCompressor` ABC; you don't need to touch the cache
or the adapter code.