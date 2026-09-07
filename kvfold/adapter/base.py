"""Adapter base class.

:class:`Adapter` is the abstract base for every framework integration
(Hugging Face, vLLM, vLLM offload). Adapters manage the lifecycle of a
framework patch:

* :meth:`attach` — install the patch (idempotent; raises on double-call).
* :meth:`detach` — uninstall the patch (idempotent).
* :meth:`state` — return a snapshot of the adapter's lifecycle state.
* :meth:`__enter__` / :meth:`__exit__` — context manager interface.

Each concrete subclass implements :meth:`attach` and :meth:`detach`. The
adapter holds a strong reference to itself on attach so it is not GC'd
before :meth:`detach` is called.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Generic, TypeVar


StateT = TypeVar("StateT")


class Adapter(AbstractContextManager, ABC, Generic[StateT]):
    """Abstract base for framework adapters."""

    name: str = "adapter"

    def __init__(self, **kwargs: Any) -> None:
        self.attached: bool = False

    @abstractmethod
    def attach(self) -> StateT:
        """Install the adapter's patch. Returns the new state snapshot."""

    @abstractmethod
    def detach(self) -> None:
        """Uninstall the adapter's patch and restore original behaviour."""

    def state(self) -> StateT:
        """Return the adapter's current state. Subclasses override."""
        raise NotImplementedError

    def __enter__(self) -> "Adapter":
        self.attach()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.detach()


__all__ = ["Adapter"]
