"""Strict, typed configuration objects for every compression method.

Every compression method in kvfold accepts a :class:`MethodConfig` instance
rather than a free-form kwargs bag. The base class defines the contract for
round-trip serialisation, registry lookup, and post-init validation. Each
concrete subclass enforces its own range and type rules.

Registry
--------
:class:`MethodConfigRegistry` is the single source of truth for which
method names are supported and how to instantiate their config from kwargs.
Adding a new method means: define a config subclass and call
``REGISTRY.register("name", NewConfig)``.

Round-trip
----------
Every concrete config supports ``to_dict()`` and ``from_dict(d)`` such that
``from_dict(c.to_dict()) == c``. The base class provides default
``to_dict`` / ``from_dict`` implementations that work for plain dataclasses;
subclasses override only when they hold non-serialisable state.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Final, Literal, Mapping, Type, TypeAlias

import torch

from kvfold.errors import MethodConfigError


Method: TypeAlias = Literal[
    "jolt", "flash", "low", "int2", "int4", "int8", "fp8", "fp16", "bf16", "pass",
]


class MethodConfig(ABC):
    """Abstract base for every method's configuration object.

    Subclasses are dataclasses. The base class is not itself a dataclass
    so the registry can enumerate subclasses without registering the base.
    """

    method: ClassVar[Method]
    registry: ClassVar["MethodConfigRegistry"]

    @abstractmethod
    def validate(self) -> None:
        """Raise :class:`MethodConfigError` if any field is out of range."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of this config."""
        return dataclasses.asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MethodConfig":
        """Construct a config from its ``to_dict()`` snapshot."""
        return cls(**dict(data))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MethodConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()


@dataclasses.dataclass(frozen=True)
class JoltConfig(MethodConfig):
    """Configuration for the JoLT compressor.

    Attributes:
        ratio: Target compression ratio (>= 1.0; 1.0 means no compression).
        bits: Candidate residual bit-widths for the allocator to choose from.
        dtype: Storage dtype for the Tucker core and bases.
        distribution: JL projection distribution (``gaussian`` or ``rademacher``).
        symmetric: If True, symmetric int quantisation around zero.
        per_channel: If True, per-channel quantisation scales.
        group_size: Optional group size for grouped quantisation; None = per-tensor.
        layer_groups: Number of layer groups the allocator treats jointly.
        seed: Seed for JL projection and SVD randomness.
    """

    method: ClassVar[Method] = "jolt"
    ratio: float = 3.0
    bits: tuple[int, ...] = (0, 2, 4, 8)
    dtype: torch.dtype = torch.float16
    distribution: Literal["gaussian", "rademacher"] = "gaussian"
    symmetric: bool = True
    per_channel: bool = True
    group_size: int | None = None
    layer_groups: int = 1
    seed: int = 0

    def validate(self) -> None:
        if self.ratio < 1.0:
            raise MethodConfigError("jolt", "ratio", f"must be >= 1.0; got {self.ratio}")
        for b in self.bits:
            if b not in (0, 2, 4, 8):
                raise MethodConfigError("jolt", "bits", f"each entry must be 0, 2, 4, or 8; got {b}")
        if self.distribution not in ("gaussian", "rademacher"):
            raise MethodConfigError("jolt", "distribution", f"unknown {self.distribution!r}")
        if self.layer_groups < 1:
            raise MethodConfigError("jolt", "layer_groups", "must be >= 1")
        if self.group_size is not None and self.group_size < 1:
            raise MethodConfigError("jolt", "group_size", "must be >= 1 when provided")


@dataclasses.dataclass(frozen=True)
class FlashConfig(MethodConfig):
    """Configuration for the FlashJoLT fast variant.

    Attributes:
        ratio: Target compression ratio (>= 1.0).
        bits: Candidate residual bit-widths.
        cap: Override the auto-derived q_cap; None means auto-compute from
            ``(length, ratio)`` via the configured :class:`CapPolicy`.
        cap_policy: Name of the cap strategy to use when ``cap`` is None.
        seed: Seed for JL projection and SVD randomness.
    """

    method: ClassVar[Method] = "flash"
    ratio: float = 3.0
    bits: tuple[int, ...] = (0, 2, 4, 8)
    cap: int | None = None
    cap_policy: Literal["linear"] = "linear"
    seed: int = 0

    def validate(self) -> None:
        if self.ratio < 1.0:
            raise MethodConfigError("flash", "ratio", f"must be >= 1.0; got {self.ratio}")
        if self.cap is not None and self.cap < 1:
            raise MethodConfigError("flash", "cap", "must be >= 1 when provided")


@dataclasses.dataclass(frozen=True)
class LowRankConfig(MethodConfig):
    """Configuration for the LowRank compressor.

    Attributes:
        rank: Token-mode rank (the feature-mode rank equals head_dim).
        dtype: Storage dtype for the factors.
        seed: Seed for SVD randomness.
    """

    method: ClassVar[Method] = "low"
    rank: int = 64
    dtype: torch.dtype = torch.float16
    seed: int = 0

    def validate(self) -> None:
        if self.rank < 1:
            raise MethodConfigError("low", "rank", f"must be >= 1; got {self.rank}")


@dataclasses.dataclass(frozen=True)
class IntConfig(MethodConfig):
    """Configuration for the IntQuant compressor (int2 / int4 / int8).

    Attributes:
        bits: Bit-width of the residual quantisation (2, 4, or 8).
        symmetric: If True, quantise symmetrically around zero.
        per_channel: If True, per-channel quantisation scales.
        group_size: Optional group size for grouped quantisation.
    """

    method: ClassVar[Method]
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = True
    group_size: int | None = None

    def validate(self) -> None:
        if self.bits not in (2, 4, 8):
            raise MethodConfigError(self.method, "bits", f"must be 2, 4, or 8; got {self.bits}")
        if self.group_size is not None and self.group_size < 1:
            raise MethodConfigError(self.method, "group_size", "must be >= 1 when provided")


@dataclasses.dataclass(frozen=True)
class FloatConfig(MethodConfig):
    """Configuration for the FloatCast compressor (fp16 / bf16).

    This is a dtype-haling passthrough — not an actual compression method.
    The name ``FloatConfig`` is deliberate: the constructor's storage dtype
    is half-precision, which costs exactly 2 bytes per element versus 4 for
    fp32, so the byte count halves. The numerical content is unchanged.

    Attributes:
        dtype: Target storage dtype; must be a 16-bit float variant.
    """

    method: ClassVar[Method]
    dtype: torch.dtype = torch.float16

    def validate(self) -> None:
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise MethodConfigError(
                self.method, "dtype", f"must be torch.float16 or torch.bfloat16; got {self.dtype}"
            )


@dataclasses.dataclass(frozen=True)
class PassConfig(MethodConfig):
    """Configuration for the Pass compressor (true passthrough).

    No transformation is applied. Useful for baselines, sanity checks, and
    "compression disabled" modes.

    Attributes:
        dtype: Optional storage dtype; if set, the tensor is cast on put.
    """

    method: ClassVar[Method] = "pass"
    dtype: torch.dtype | None = None

    def validate(self) -> None:
        if self.dtype is not None and not self.dtype.is_floating_point:
            raise MethodConfigError("pass", "dtype", "must be a floating-point dtype when provided")


@dataclasses.dataclass(frozen=True)
class Float8Config(MethodConfig):
    """Configuration for the Float8 compressor.

    Real IEEE-style FP8 quantisation with per-tensor or per-channel scale.
    Unlike :class:`FloatConfig`, this is an actual lossy compressor with
    a documented round-trip error bound.

    Attributes:
        variant: FP8 variant — ``e4m3`` (range, dynamic range ~240) or
            ``e5m2`` (wider exponent, less precision).
        per_channel: If True, per-channel quantisation scales.
        group_size: Optional group size for grouped quantisation.
    """

    method: ClassVar[Method] = "fp8"
    variant: Literal["e4m3", "e5m2"] = "e4m3"
    per_channel: bool = True
    group_size: int | None = None

    def validate(self) -> None:
        if self.variant not in ("e4m3", "e5m2"):
            raise MethodConfigError("fp8", "variant", f"unknown {self.variant!r}")
        if self.group_size is not None and self.group_size < 1:
            raise MethodConfigError("fp8", "group_size", "must be >= 1 when provided")


class Int2Config(IntConfig):
    method: ClassVar[Method] = "int2"
    bits: int = 2


class Int4Config(IntConfig):
    method: ClassVar[Method] = "int4"
    bits: int = 4


class Int8Config(IntConfig):
    method: ClassVar[Method] = "int8"
    bits: int = 8


class Fp16Config(FloatConfig):
    method: ClassVar[Method] = "fp16"
    dtype: torch.dtype = torch.float16


class Bf16Config(FloatConfig):
    method: ClassVar[Method] = "bf16"
    dtype: torch.dtype = torch.bfloat16


class MethodConfigRegistry:
    """Registry of supported :class:`MethodConfig` subclasses.

    A method is registered by binding a name to its config class. The
    :meth:`build` method is the single entry point used by the dispatcher
    to turn ``(method, kwargs)`` into a validated :class:`MethodConfig`.

    Registrations are append-only; the same name cannot be bound twice.
    """

    def __init__(self) -> None:
        self.entries: dict[Method, Type[MethodConfig]] = {}

    def register(self, name: Method, config_cls: Type[MethodConfig]) -> None:
        """Bind ``name`` to ``config_cls``.

        Raises:
            ValueError: If ``name`` is already registered.
        """
        if name in self.entries:
            raise ValueError(f"method {name!r} is already registered to {self.entries[name].__name__}")
        if not issubclass(config_cls, MethodConfig):
            raise TypeError(f"{config_cls.__name__} must inherit from MethodConfig")
        config_cls.registry = self  # type: ignore[attr-defined]
        config_cls.method = name  # type: ignore[attr-defined]
        self.entries[name] = config_cls

    def names(self) -> tuple[Method, ...]:
        """Return the registered method names in insertion order."""
        return tuple(self.entries)

    def resolve(self, name: str) -> Type[MethodConfig]:
        """Return the config class bound to ``name``.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        try:
            return self.entries[name]  # type: ignore[return-value]
        except KeyError:
            raise KeyError(f"unknown method {name!r}; registered: {self.names()}") from None

    def build(self, name: str, **kwargs: Any) -> MethodConfig:
        """Construct a validated config from ``(name, kwargs)``.

        Unknown kwargs raise :class:`MethodConfigError` with the offending
        field name.
        """
        config_cls = self.resolve(name)
        valid_fields = {f.name for f in dataclasses.fields(config_cls)}
        unknown = set(kwargs) - valid_fields
        if unknown:
            raise MethodConfigError(
                name,
                ",".join(sorted(unknown)),
                f"unknown field(s); valid: {sorted(valid_fields)}",
            )
        config = config_cls(**kwargs)
        config.validate()
        return config

    def supported(self) -> tuple[Method, ...]:
        """Return the registered method names (alias for :meth:`names`)."""
        return self.names()


REGISTRY: Final[MethodConfigRegistry] = MethodConfigRegistry()
"""Process-wide registry; populated by module import side effects."""


def register_method(name: Method, config_cls: Type[MethodConfig]) -> Type[MethodConfig]:
    """Decorator-style helper for class definition sites.

    Usage::

        @register_method("jolt", JoltConfig)
        @dataclasses.dataclass(frozen=True)
        class JoltConfig(MethodConfig):
            ...
    """
    REGISTRY.register(name, config_cls)
    return config_cls


REGISTRY.register("jolt", JoltConfig)
REGISTRY.register("flash", FlashConfig)
REGISTRY.register("low", LowRankConfig)
REGISTRY.register("int2", Int2Config)
REGISTRY.register("int4", Int4Config)
REGISTRY.register("int8", Int8Config)
REGISTRY.register("fp8", Float8Config)
REGISTRY.register("fp16", Fp16Config)
REGISTRY.register("bf16", Bf16Config)
REGISTRY.register("pass", PassConfig)


__all__ = [
    "Method",
    "MethodConfig",
    "MethodConfigRegistry",
    "JoltConfig",
    "FlashConfig",
    "LowRankConfig",
    "IntConfig",
    "Int2Config",
    "Int4Config",
    "Int8Config",
    "FloatConfig",
    "Float8Config",
    "Fp16Config",
    "Bf16Config",
    "PassConfig",
    "REGISTRY",
    "register_method",
]
