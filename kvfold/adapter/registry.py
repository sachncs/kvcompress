"""Family registry — collapses the 9 per-family shims into one module.

Each :class:`Family` instance owns the install/uninstall logic for one
HF ``config.model_type``. The :class:`FamilyRegistry` is a single
mapping ``model_type -> Family``. Adding a new family means
``@register("name")`` decorating a :class:`Family` subclass.

Today every entry is a no-op :class:`NoOpFamily` because the HF
:class:`~transformers.cache_utils.DynamicCache` subclass already
covers the standard cache layout. The registry exists so future
model-specific hooks (custom attention kernels, MLA, fused QKV) have a
place to land without scattering no-op modules across the package.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Type

log = logging.getLogger(__name__)


class Family(ABC):
    """Base class for a per-model-family install/uninstall policy."""

    name: str

    @abstractmethod
    def install(self, model: object, pool: object) -> Callable[[], None] | None:
        """Install the family-specific hooks on ``model``."""

    def uninstall(self, model: object, pool: object) -> None:
        """Inverse of :meth:`install`. Default: no-op."""


class NoOpFamily(Family):
    """Family whose install is a no-op (the HF cache subclass covers it)."""

    name: str = ""

    def install(self, model: object, pool: object) -> Callable[[], None] | None:
        return None


class Llama(NoOpFamily):
    name = "llama"


class Mistral(NoOpFamily):
    name = "mistral"


class Qwen2(NoOpFamily):
    name = "qwen2"


class Qwen2Moe(NoOpFamily):
    name = "qwen2_moe"


class Gemma(NoOpFamily):
    name = "gemma"


class Gemma2(NoOpFamily):
    name = "gemma2"


class Phi(NoOpFamily):
    name = "phi"


class Phi3(NoOpFamily):
    name = "phi3"


class Mixtral(NoOpFamily):
    name = "mixtral"


class Falcon(NoOpFamily):
    name = "falcon"


class DeepSeek(NoOpFamily):
    name = "deepseek"


class InternLM(NoOpFamily):
    name = "internlm"


class FamilyRegistry:
    """Registry of :class:`Family` strategies keyed by HF ``model_type``."""

    def __init__(self) -> None:
        self.entries: dict[str, Family] = {}

    def register(self, family_cls: Type[Family]) -> Type[Family]:
        """Bind ``family_cls.name`` to a fresh instance of ``family_cls``.

        Idempotent: re-registering the same class under the same ``name``
        is a no-op so ``importlib.reload`` and Jupyter re-import paths
        do not blow up. Registering a different class under the same
        ``name`` still raises.
        """
        if not family_cls.name:
            raise ValueError(f"{family_cls.__name__} must set the `name` class attribute")
        if family_cls.name in self.entries:
            if type(self.entries[family_cls.name]) is not family_cls:
                raise ValueError(f"family {family_cls.name!r} is already registered")
            return family_cls
        self.entries[family_cls.name] = family_cls()
        return family_cls

    def names(self) -> tuple[str, ...]:
        return tuple(self.entries)

    def resolve(self, model_type: str) -> Family | None:
        return self.entries.get(model_type)

    def install(self, model: object, pool: object, model_type: str) -> Callable[[], None] | None:
        family = self.resolve(model_type)
        if family is None:
            log.warning(
                "kvfold: no family shim for model_type=%s; using generic interception",
                model_type,
            )
            return None
        return family.install(model, pool)


REGISTRY: FamilyRegistry = FamilyRegistry()


def known_model_types() -> list[str]:
    """Sorted list of all registered ``model_type`` strings."""
    return sorted(REGISTRY.names())


def register(family_cls: Type[Family]) -> Type[Family]:
    """Module-level decorator helper for family registration."""
    return REGISTRY.register(family_cls)


def resolve(model_type: str) -> Family | None:
    """Return the family handling ``model_type`` or ``None``."""
    return REGISTRY.resolve(model_type)


def install(model: object, pool: object, model_type: str) -> Callable[[], None] | None:
    """Dispatch to the right family and invoke its install."""
    return REGISTRY.install(model, pool, model_type)


def _register_builtins() -> None:
    for cls in (Llama, Mistral, Qwen2, Qwen2Moe, Gemma, Gemma2, Phi, Phi3, Mixtral, Falcon, DeepSeek, InternLM):
        REGISTRY.register(cls)


_register_builtins()


__all__ = [
    "Family",
    "NoOpFamily",
    "FamilyRegistry",
    "REGISTRY",
    "register",
    "known_model_types",
    "resolve",
    "install",
    "Llama",
    "Mistral",
    "Qwen2",
    "Qwen2Moe",
    "Gemma",
    "Gemma2",
    "Phi",
    "Phi3",
    "Mixtral",
    "Falcon",
    "DeepSeek",
    "InternLM",
]
