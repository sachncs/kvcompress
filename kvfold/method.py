"""Method names and helpers.

The :data:`Method` literal is the canonical list of supported compression
method names. The module also exposes helpers for family classification
(quant, low, pass), GPU requirement checks, and case normalisation.
"""

from __future__ import annotations

from typing import Final, Literal, TypeGuard


Method: TypeAlias = Literal[
    "jolt", "flash", "low", "int2", "int4", "int8", "fp8", "fp16", "bf16", "pass",
]
"""Canonical compression-method names accepted by the dispatcher."""


METHODS: Final[tuple[Method, ...]] = (
    "jolt", "flash", "low", "int2", "int4", "int8", "fp8", "fp16", "bf16", "pass",
)
"""Tuple of every supported method name in insertion order."""


QUANT_METHODS: Final[frozenset[Method]] = frozenset({"int2", "int4", "int8", "fp8"})
"""Methods that store a per-element quantised payload."""


LOW_METHODS: Final[frozenset[Method]] = frozenset({"jolt", "flash", "low"})
"""Methods that store a low-rank factorisation."""


DTYPE_METHODS: Final[frozenset[Method]] = frozenset({"fp16", "bf16"})
"""Methods that only change storage dtype — no real compression."""


PASSTHROUGH_METHODS: Final[frozenset[Method]] = frozenset({"pass"})
"""Methods that copy through the input unchanged."""


def family(name: Method) -> Literal["quant", "low", "dtype", "pass"]:
    """Return the algorithm family a method belongs to.

    Raises:
        ValueError: If ``name`` is not a registered method.
    """
    if name in LOW_METHODS:
        return "low"
    if name in QUANT_METHODS:
        return "quant"
    if name in DTYPE_METHODS:
        return "dtype"
    if name in PASSTHROUGH_METHODS:
        return "pass"
    raise ValueError(f"unknown method {name!r}")


def is_quant(name: Method) -> TypeGuard[Literal["int2", "int4", "int8", "fp8"]]:
    """True if the method produces a per-element quantised payload."""
    return name in QUANT_METHODS  # type: ignore[return-value]


def is_low(name: Method) -> TypeGuard[Literal["jolt", "flash", "low"]]:
    """True if the method stores a low-rank factorisation."""
    return name in LOW_METHODS  # type: ignore[return-value]


def is_dtype_only(name: Method) -> TypeGuard[Literal["fp16", "bf16"]]:
    """True if the method only changes storage dtype (no real compression)."""
    return name in DTYPE_METHODS  # type: ignore[return-value]


def is_passthrough(name: Method) -> TypeGuard[Literal["pass"]]:
    """True if the method copies the input through unchanged."""
    return name == "pass"


def requires_gpu(name: Method) -> bool:
    """True if the method's fastest path requires a GPU.

    All methods run on CPU; this flag exists so a deployment can warn
    when an expected-GPU method is forced onto CPU.
    """
    return name in ("jolt", "flash")


def normalise(name: str) -> Method:
    """Lower-case ``name`` and reject anything not in :data:`METHODS`.

    Raises:
        ValueError: If ``name`` is not a registered method.
    """
    candidate = name.strip().lower()
    if candidate in METHODS:  # type: ignore[operator]
        return candidate  # type: ignore[return-value]
    raise ValueError(f"unknown method {name!r}; supported: {METHODS}")


__all__ = [
    "Method",
    "METHODS",
    "QUANT_METHODS",
    "LOW_METHODS",
    "DTYPE_METHODS",
    "PASSTHROUGH_METHODS",
    "family",
    "is_quant",
    "is_low",
    "is_dtype_only",
    "is_passthrough",
    "requires_gpu",
    "normalise",
]
