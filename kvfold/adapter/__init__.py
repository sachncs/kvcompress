"""Serving-framework adapters and the Hugging Face entry point."""

from __future__ import annotations

from typing import Any

from kvfold.adapter.base import Adapter, AdapterState


def __getattr__(name: str) -> Any:
    if name == "HF":
        from kvfold.adapter.huggingface import HF

        return HF
    if name in ("export_kv", "import_kv", "is_vllm_available", "resolve_cache"):
        from kvfold.adapter import vllm

        return getattr(vllm, name)
    if name in ("Offload", "EvictPool", "is_vllm_offload_available", "vllm_offload_version"):
        from kvfold.adapter import vllm_offload

        return getattr(vllm_offload, name)
    raise AttributeError(f"module 'kvfold.adapter' has no attribute {name!r}")


__all__ = [
    "HF",
    "Adapter",
    "AdapterState",
    "EvictPool",
    "Offload",
    "export_kv",
    "import_kv",
    "is_vllm_available",
    "is_vllm_offload_available",
    "resolve_cache",
    "vllm_offload_version",
]
