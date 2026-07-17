"""Unified compressor dispatch.

Public surface: :func:`build_compressor` is the single source of truth for
mapping the ``method`` string in :func:`kvcompress.api.enable_compression`
to a concrete :class:`~kvcompress.compressor.base.KVCompressor` subclass.

Why one place: the same dispatch was duplicated across
``adapters/huggingface.py`` and ``adapters/vllm.py`` and only handled three
methods (``jolt``, ``flashjolt``, ``identity``) while the public API and
README advertised nine (``int2``, ``int4``, ``int8``, ``fp8``, ``fp16``,
``bf16``, ``lowrank``, ``jolt``, ``flashjolt``, ``identity``). Collapsing
the duplication closes the doc/code drift.

Per-method supported kwargs are documented inline. The dispatch filters
unknown kwargs at the call site rather than forwarding them to the
compressor constructor — a typo (``per_chanel=True`` instead of
``per_channel=True``) becomes a clear error instead of a silent drop.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from kvcompress.compressor.base import KVCompressor

__all__ = ["METHODS", "build_compressor", "supported_methods"]


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-method keyword argument allow-list
# ---------------------------------------------------------------------------
#
# Only the kwargs listed here are forwarded to the compressor constructor.
# ``enable_compression(**kwargs)`` rejects unknown kwargs with a clear
# error message at the API boundary so a typo (e.g. ``per_chanel=True``)
# fails fast rather than being silently dropped.
# ---------------------------------------------------------------------------

INT_KWARGS = frozenset(
    {
        "bits",
        "per_channel",
        "symmetric",
        "group_size",
        "factor_dtype",
    }
)

FLOAT_KWARGS = frozenset({"factor_dtype"})

LOWRANK_KWARGS = frozenset({"rank", "factor_dtype", "seed"})

JOLT_KWARGS = frozenset(
    {
        "compression_ratio",
        "bits",
        "factor_dtype",
        "jl_distribution",
        "symmetric_quant",
        "per_channel_quant",
        "group_size",
        "layer_groups",
        "seed",
    }
)

IDENTITY_KWARGS = frozenset({"factor_dtype"})

# ---------------------------------------------------------------------------
# Method catalog
# ---------------------------------------------------------------------------

METHODS: dict[str, dict[str, Any]] = {
    "jolt": {
        "class_path": ("kvcompress.compressor.jolt", "JoLTCompressor"),
        "allowed_kwargs": JOLT_KWARGS,
    },
    "flashjolt": {
        "class_path": ("kvcompress.compressor.flashjolt", "FlashJoLTCompressor"),
        "allowed_kwargs": JOLT_KWARGS,
    },
    "lowrank": {
        "class_path": ("kvcompress.compressor.lowrank", "LowRankCompressor"),
        "allowed_kwargs": LOWRANK_KWARGS,
    },
    "int2": {
        "class_path": ("kvcompress.compressor.quantization_only", "IntQuantOnlyCompressor"),
        "allowed_kwargs": INT_KWARGS,
        "bits_override": 2,
    },
    "int4": {
        "class_path": ("kvcompress.compressor.quantization_only", "IntQuantOnlyCompressor"),
        "allowed_kwargs": INT_KWARGS,
        "bits_override": 4,
    },
    "int8": {
        "class_path": ("kvcompress.compressor.quantization_only", "IntQuantOnlyCompressor"),
        "allowed_kwargs": INT_KWARGS,
        "bits_override": 8,
    },
    "fp8": {
        "class_path": ("kvcompress.compressor.identity", "IdentityCompressor"),
        "allowed_kwargs": IDENTITY_KWARGS,
        "dtype_override": torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16,
    },
    "fp16": {
        "class_path": ("kvcompress.compressor.identity", "IdentityCompressor"),
        "allowed_kwargs": IDENTITY_KWARGS,
        "dtype_override": torch.float16,
    },
    "bf16": {
        "class_path": ("kvcompress.compressor.identity", "IdentityCompressor"),
        "allowed_kwargs": IDENTITY_KWARGS,
        "dtype_override": torch.bfloat16,
    },
    "identity": {
        "class_path": ("kvcompress.compressor.identity", "IdentityCompressor"),
        "allowed_kwargs": IDENTITY_KWARGS,
    },
}


def supported_methods() -> tuple[str, ...]:
    """Tuple of every compression method the dispatcher accepts."""
    return tuple(METHODS.keys())


def build_compressor(method: str, **kwargs: Any) -> KVCompressor:
    """Construct a compressor from its public method name.

    Args:
        method: one of :func:`supported_methods`.
        **kwargs: forwarded to the compressor constructor. Unknown kwargs
            raise :class:`ValueError` immediately so a typo is loud.

    Returns:
        A fresh :class:`KVCompressor` instance.

    Raises:
        ValueError: if ``method`` is unknown, or if any kwarg isn't on
            the method's allow-list.
    """
    import importlib

    method = method.lower()
    spec = METHODS.get(method)
    if spec is None:
        supported = ", ".join(repr(m) for m in METHODS)
        raise NotImplementedError(
            f"compressor method {method!r} is not supported; supported methods: {supported}."
        )

    allowed = spec["allowed_kwargs"]
    unknown = set(kwargs) - allowed - {"compression_ratio", "bits", "seed", "layer_groups"}
    if unknown:
        raise ValueError(
            f"compressor method {method!r} got unexpected kwargs: "
            f"{sorted(unknown)!r}; supported: {sorted(allowed)!r}."
        )

    if "bits_override" in spec:
        kwargs.setdefault("bits", spec["bits_override"])
    if "dtype_override" in spec:
        kwargs.setdefault("factor_dtype", spec["dtype_override"])

    module_name, attr = spec["class_path"]
    cls = getattr(importlib.import_module(module_name), attr)
    result = cls(**kwargs)
    # ponytail: cast away the Any introduced by `**kwargs`.
    if not isinstance(result, KVCompressor):
        raise TypeError(f"{cls.__name__} did not return a KVCompressor")
    return result
