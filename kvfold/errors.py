"""Error hierarchy for kvfold.

All exceptions raised by the library inherit from :class:`KVCompressError`,
which itself inherits from :class:`Exception`. Each subclass carries a
machine-readable :attr:`code` and an optional human-readable :attr:`hint`
describing how to resolve the failure.

The hierarchy is intentionally narrow so callers can ``except
KVCompressError`` to catch every library-level failure, or target a specific
subclass for fine-grained handling.

Classes
-------
KVCompressError
    Base class for all library-level exceptions.

CacheError
    Cache storage failures.

CacheValidationError
    Payload shape or dtype mismatch on a ``put``.

CacheMissingLayerError
    Requested layer is not present in the cache.

CacheShapeError
    Per-layer shape contract violated.

AllocatorError
    Byte-budget allocation failures.

AllocatorNoFeasibleError
    Candidate grid is too coarse for the requested ratio.

AllocatorInvalidTargetError
    Target ratio is outside the supported range.

VLLMError
    vLLM integration failures.

VLLMNotAvailableError
    vLLM is not importable in the current environment.

VLLMAPIDriftError
    Installed vLLM does not expose the expected API surface.

MethodError
    Compressor-method dispatch failures.

UnsupportedMethodError
    Requested compression method is not registered.

MethodConfigError
    Method configuration is invalid.

ShapeError
    Tensor shape contract violated.

DTypeError
    Tensor dtype is not supported.

DeviceError
    Tensor device is not supported.

KernelError
    Acceleration kernel failures.

KernelNotAvailableError
    Requested kernel backend is not available in this environment.
"""

from __future__ import annotations

from typing import Final


class KVCompressError(Exception):
    """Base class for every error raised by kvfold."""

    code: str = "kvfold_error"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint is None:
            return self.message
        return f"{self.message} (hint: {self.hint})"

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serialisable representation of this error."""
        return {"code": self.code, "message": self.message, "hint": self.hint}


class CacheError(KVCompressError):
    """Cache storage failures."""

    code: Final[str] = "cache_error"


class CacheValidationError(CacheError):
    """Payload shape or dtype does not match the cache contract on put."""

    code: Final[str] = "cache_validation_error"

    def __init__(self, layer: int, expected: tuple[int, ...], actual: tuple[int, ...]) -> None:
        super().__init__(
            f"layer {layer}: expected shape {tuple(expected)}, got {tuple(actual)}",
            hint="check that the tensor passed to put() matches the cache layout",
        )
        self.layer = layer
        self.expected = tuple(expected)
        self.actual = tuple(actual)


class CacheMissingLayerError(CacheError):
    """Requested layer is not present in the cache."""

    code: Final[str] = "cache_missing_layer_error"

    def __init__(self, layer: int) -> None:
        super().__init__(f"layer {layer} not present in cache")
        self.layer = layer


class CacheShapeError(CacheError):
    """Per-layer shape contract violated."""

    code: Final[str] = "cache_shape_error"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)


class AllocatorError(KVCompressError):
    """Byte-budget allocation failures."""

    code: Final[str] = "allocator_error"


class AllocatorNoFeasibleError(AllocatorError):
    """Candidate grid is too coarse for the requested compression ratio."""

    code: Final[str] = "allocator_no_feasible_error"

    def __init__(self, target_ratio: float, achieved_ratio: float) -> None:
        super().__init__(
            f"target ratio {target_ratio:.3f} not reachable; best achievable {achieved_ratio:.3f}",
            hint="widen the candidate grid (more bits, more token ranks) or lower the target ratio",
        )
        self.target_ratio = target_ratio
        self.achieved_ratio = achieved_ratio


class AllocatorInvalidTargetError(AllocatorError):
    """Target ratio is outside the supported range."""

    code: Final[str] = "allocator_invalid_target_error"

    def __init__(self, target_ratio: float, *, minimum: float = 1.0) -> None:
        super().__init__(
            f"target ratio {target_ratio:.3f} must be >= {minimum:.3f}",
            hint="compression ratios below 1.0 mean expansion, not compression",
        )
        self.target_ratio = target_ratio
        self.minimum = minimum


class VLLMError(KVCompressError):
    """vLLM integration failures."""

    code: Final[str] = "vllm_error"


class VLLMNotAvailableError(VLLMError):
    """vLLM is not importable in the current environment."""

    code: Final[str] = "vllm_not_available_error"

    def __init__(self) -> None:
        super().__init__(
            "vLLM is not importable",
            hint="install with `pip install kvfold[vllm]`",
        )


class VLLMAPIDriftError(VLLMError):
    """Installed vLLM does not expose the expected API surface."""

    code: Final[str] = "vllm_api_drift_error"

    def __init__(self, missing: str) -> None:
        super().__init__(
            f"vLLM API drift: expected attribute {missing!r} not found",
            hint="upgrade vLLM or downgrade kvfold to a compatible version",
        )
        self.missing = missing


class MethodError(KVCompressError):
    """Compressor-method dispatch failures."""

    code: Final[str] = "method_error"


class UnsupportedMethodError(MethodError):
    """Requested compression method is not registered."""

    code: Final[str] = "unsupported_method_error"

    def __init__(self, method: str, supported: tuple[str, ...]) -> None:
        super().__init__(
            f"unknown method {method!r}; supported: {', '.join(supported)}",
            hint=f"pick one of {list(supported)} or register a custom compressor",
        )
        self.method = method
        self.supported = tuple(supported)


class MethodConfigError(MethodError):
    """Method configuration is invalid."""

    code: Final[str] = "method_config_error"

    def __init__(self, method: str, field: str, message: str) -> None:
        super().__init__(
            f"invalid config for method {method!r}: {field}: {message}",
            hint=f"see kvfold.config.{method.title()}Config for valid values",
        )
        self.method = method
        self.field = field
        self.detail = message


class ShapeError(KVCompressError):
    """Tensor shape contract violated."""

    code: Final[str] = "shape_error"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)


class DTypeError(KVCompressError):
    """Tensor dtype is not supported in this context."""

    code: Final[str] = "dtype_error"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)


class DeviceError(KVCompressError):
    """Tensor device is not supported in this context."""

    code: Final[str] = "device_error"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)


class KernelError(KVCompressError):
    """Acceleration kernel failures."""

    code: Final[str] = "kernel_error"


class KernelNotAvailableError(KernelError):
    """Requested kernel backend is not available in this environment."""

    code: Final[str] = "kernel_not_available_error"

    def __init__(self, backend: str, required: str) -> None:
        super().__init__(
            f"kernel backend {backend!r} requires {required}, which is not available",
            hint=f"install {required} or switch to backend='torch'",
        )
        self.backend = backend
        self.required = required
