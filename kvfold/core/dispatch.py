"""Unified compressor dispatch and registry.

:class:`CompressorRegistry` is the single source of truth for mapping a
method string to a concrete :class:`Compressor` subclass and its
:class:`MethodConfig`. Each registered entry binds:

* a public method name (``"jolt"``, ``"flash"``, ...),
* the :class:`Compressor` subclass that implements it,
* the :class:`MethodConfig` subclass that configures it,
* an optional kwarg adapter for backwards-friendly defaults.

Adding a new method means: subclass :class:`Compressor` and
:class:`MethodConfig`, then call :meth:`CompressorRegistry.register`.

The legacy :func:`build_compressor` function is a thin façade that calls
:func:`kvfold.api.build_compressor`; users should import from the public
``kvfold`` namespace.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
from typing import Any, Callable, Mapping, Type

from kvfold.config import MethodConfig, REGISTRY as CONFIG_REGISTRY
from kvfold.core.base import Compressor
from kvfold.errors import MethodConfigError, UnsupportedMethodError

log = logging.getLogger(__name__)


class CompressorEntry:
    """Binding between a method name, its config class, and its compressor class.

    The :attr:`factory` callable is the adapter that turns a
    :class:`MethodConfig` into a configured :class:`Compressor` instance.
    It is computed once at registration time and cached.
    """

    def __init__(
        self,
        method: str,
        config_cls: Type[MethodConfig],
        compressor_cls: Type[Compressor],
        factory: Callable[[MethodConfig], Compressor],
    ) -> None:
        self.method = method
        self.config_cls = config_cls
        self.compressor_cls = compressor_cls
        self.factory = factory


class CompressorRegistry:
    """Process-wide registry of compression methods.

    The KV store maps method name → :class:`CompressorEntry`. Registration
    is append-only; the same name cannot be bound twice.
    """

    def __init__(self) -> None:
        self.entries: dict[str, CompressorEntry] = {}

    def register(
        self,
        method: str,
        compressor_cls: Type[Compressor],
        factory: Callable[[MethodConfig], Compressor] | None = None,
        config_cls: Type[MethodConfig] | None = None,
    ) -> Type[Compressor]:
        """Bind ``method`` to ``compressor_cls``.

        Args:
            method: public method name (e.g. ``"jolt"``).
            compressor_cls: subclass of :class:`Compressor`.
            factory: optional callable ``(MethodConfig) -> Compressor``.
                If omitted, the default factory uses :meth:`Compressor.from_config`.
            config_cls: optional :class:`MethodConfig` subclass. If omitted,
                the currently-registered config in :data:`CONFIG_REGISTRY`
                is used.

        Returns:
            The ``compressor_cls`` argument (for use as a decorator).
        """
        if method in self.entries:
            raise ValueError(f"method {method!r} is already registered to {self.entries[method].compressor_cls.__name__}")
        if not issubclass(compressor_cls, Compressor):
            raise TypeError(f"{compressor_cls.__name__} must inherit from Compressor")

        if config_cls is None:
            try:
                config_cls = CONFIG_REGISTRY.resolve(method)
            except KeyError as exc:
                raise MethodConfigError(method, "config", f"no MethodConfig registered for {method!r}") from exc

        if factory is None:
            config_fields = {f.name for f in dataclasses.fields(config_cls)}
            compressor_params = compressor_cls.__init__.__code__.co_varnames

            def default_factory(config: MethodConfig) -> Compressor:
                config_dict = dataclasses.asdict(config)
                kwargs = {k: v for k, v in config_dict.items() if k in config_fields and k in compressor_params}
                return compressor_cls(**kwargs)

            factory = default_factory

        entry = CompressorEntry(
            method=method,
            config_cls=config_cls,
            compressor_cls=compressor_cls,
            factory=factory,
        )
        self.entries[method] = entry
        return compressor_cls

    def names(self) -> tuple[str, ...]:
        """Return the registered method names in insertion order."""
        return tuple(self.entries)

    def resolve(self, method: str) -> CompressorEntry:
        """Return the entry for ``method`` or raise :class:`UnsupportedMethodError`."""
        try:
            return self.entries[method]
        except KeyError:
            raise UnsupportedMethodError(method=method, supported=self.names()) from None

    def build(self, method: str, **kwargs: Any) -> Compressor:
        """Build a configured :class:`Compressor` from kwargs.

        The kwargs are validated against the method's :class:`MethodConfig`,
        then the entry's factory is invoked.
        """
        config = CONFIG_REGISTRY.build(method, **kwargs)
        entry = self.resolve(method)
        compressor = entry.factory(config)
        if not isinstance(compressor, Compressor):
            raise TypeError(
                f"factory for {method!r} returned {type(compressor).__name__}, expected Compressor"
            )
        return compressor


REGISTRY: CompressorRegistry = CompressorRegistry()
"""Process-wide registry; populated by module import side effects."""


def register(
    method: str,
    compressor_cls: Type[Compressor],
    factory: Callable[[MethodConfig], Compressor] | None = None,
) -> Type[Compressor]:
    """Module-level decorator helper for compressor registration.

    Usage::

        @register("jolt", Jolt)
        class Jolt(Compressor):
            ...
    """
    return REGISTRY.register(method, compressor_cls, factory)


__all__ = [
    "CompressorEntry",
    "CompressorRegistry",
    "REGISTRY",
    "register",
]
