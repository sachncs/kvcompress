"""Tests for the lazy-export ``__getattr__`` in adapters/__init__.py.

The adapters subpackage exposes HuggingFaceAdapter, vLLM helpers, and
the OffloadingHandler through a single ``__getattr__`` so the import
chain is shallow and missing optional deps don't blow up the package
import. These tests cover the dispatch path.
"""

from __future__ import annotations

import pytest


def test_huggingface_adapter_lazy_export() -> None:
    from kvcompress.adapters import HuggingFaceAdapter

    assert HuggingFaceAdapter.__name__ == "HuggingFaceAdapter"


def test_vllm_lazy_exports() -> None:
    from kvcompress.adapters import (
        export_kv,
        import_kv,
        is_vllm_available,
        resolve_cache,
    )

    assert callable(export_kv)
    assert callable(import_kv)
    assert callable(is_vllm_available)
    assert callable(resolve_cache)


def test_vllm_kv_offload_lazy_exports() -> None:
    from kvcompress.adapters import (
        JoLTOffloadHandler,
        ThreadSafeEvictionPool,
        is_vllm_kv_offload_available,
    )

    assert JoLTOffloadHandler.__name__ == "JoLTOffloadHandler"
    assert ThreadSafeEvictionPool.__name__ == "ThreadSafeEvictionPool"
    assert callable(is_vllm_kv_offload_available)


def test_unknown_attribute_raises() -> None:
    import kvcompress.adapters as sub

    with pytest.raises(AttributeError, match="no attribute"):
        _ = sub.NoSuchSymbol


def test_lazy_exports_cached_after_first_lookup() -> None:
    """After the first ``__getattr__`` resolution, the symbol is cached."""
    import kvcompress.adapters as sub

    a = sub.HuggingFaceAdapter
    b = sub.HuggingFaceAdapter
    assert a is b  # cached reference, same object
