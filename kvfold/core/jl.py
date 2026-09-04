"""Johnson-Lindenstrauss random projections.

A :class:`Projector` maps a high-dimensional vector ``x`` (length
``in_dim``) to a low-dimensional vector ``y = x @ matrix.T`` (length
``out_dim``) where ``matrix`` is a random sign-or-Gaussian matrix. The
:class:`Projector` ABC defines the contract; concrete strategies are
:class:`Gaussian`, :class:`Rademacher`, and :class:`Sparse`.

The approximate inverse ``x̂ = y @ matrix`` is *not* the true inverse
of the projection — it is the cheap JL reconstruction the paper uses,
with error bounded by the JL lemma.

Caching
-------
:class:`ProjectionCache` memoises projections by
``(out_dim, in_dim, distribution, seed, device, dtype)`` so repeated
calls with the same shape + seed reuse the matrix.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch

from kvfold.errors import DTypeError, ShapeError

__all__ = [
    "Projector",
    "Gaussian",
    "Rademacher",
    "Sparse",
    "Projection",
    "ProjectionCache",
    "projection",
    "projection_cache_info",
    "clear_projection_cache",
]


log = logging.getLogger(__name__)


Distribution = Literal["gaussian", "rademacher", "sparse"]


@dataclass
class Projection:
    """A random projection matrix and its metadata.

    Attributes:
        matrix: the ``(out_dim, in_dim)`` projection matrix.
        distribution: name of the strategy used.
        seed: seed used to construct the matrix.
    """

    matrix: torch.Tensor
    distribution: Distribution
    seed: int

    @property
    def device(self) -> torch.device:
        return self.matrix.device

    @property
    def dtype(self) -> torch.dtype:
        return self.matrix.dtype

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` to ``y = x @ matrix.T``.

        ``x`` must have shape ``(..., in_dim)``.
        """
        if x.shape[-1] != self.matrix.shape[1]:
            raise ShapeError(
                f"input trailing dim {x.shape[-1]} != projector in_dim {self.matrix.shape[1]}",
                hint="pass a tensor whose last dim matches the projection's input dim",
            )
        return x @ self.matrix.t()

    def apply_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Cheap JL inverse ``x̂ = x @ matrix``.

        This is *not* the true matrix inverse; it is the paper's
        approximation, with error bounded by the JL lemma.
        """
        return x @ self.matrix


class Projector(ABC):
    """Strategy for constructing a random projection matrix."""

    distribution: Distribution

    @abstractmethod
    def build(self, out_dim: int, in_dim: int, *, seed: int, device: torch.device | str, dtype: torch.dtype) -> Projection: ...


class Gaussian(Projector):
    """Gaussian JL projection; entries drawn from ``N(0, 1/in_dim)``."""

    distribution: Distribution = "gaussian"

    def build(self, out_dim: int, in_dim: int, *, seed: int, device: torch.device | str, dtype: torch.dtype) -> Projection:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        matrix = torch.randn(int(out_dim), int(in_dim), generator=gen, dtype=torch.float32) / (in_dim ** 0.5)
        return Projection(matrix=matrix.to(device=device, dtype=dtype), distribution="gaussian", seed=int(seed))


class Rademacher(Projector):
    """Rademacher JL projection; entries are independent ±1."""

    distribution: Distribution = "rademacher"

    def build(self, out_dim: int, in_dim: int, *, seed: int, device: torch.device | str, dtype: torch.dtype) -> Projection:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        sign_matrix = (torch.randint(0, 2, (int(out_dim), int(in_dim)), generator=gen) * 2 - 1).to(torch.float32)
        scale = 1.0 / (in_dim ** 0.5)
        matrix = sign_matrix * scale
        return Projection(matrix=matrix.to(device=device, dtype=dtype), distribution="rademacher", seed=int(seed))


class Sparse(Projector):
    """Sparse JL projection; each column has ``sparsity`` non-zero Rademacher entries."""

    distribution: Distribution = "sparse"

    def __init__(self, sparsity: float = 0.1) -> None:
        if not 0 < sparsity <= 1:
            raise ValueError(f"sparsity must be in (0, 1]; got {sparsity}")
        self.sparsity = float(sparsity)

    def build(self, out_dim: int, in_dim: int, *, seed: int, device: torch.device | str, dtype: torch.dtype) -> Projection:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        matrix = torch.zeros(int(out_dim), int(in_dim), dtype=torch.float32)
        n_per_col = max(1, int(round(self.sparsity * in_dim)))
        scale = (1.0 / (n_per_col ** 0.5)) / (in_dim ** 0.5) ** 0  # JL Achlioptas factor
        for col in range(int(in_dim)):
            rows = torch.randperm(int(out_dim), generator=gen)[:n_per_col]
            signs = (torch.randint(0, 2, (n_per_col,), generator=gen) * 2 - 1).to(torch.float32)
            matrix[rows, col] = signs
        matrix = matrix * scale
        return Projection(matrix=matrix.to(device=device, dtype=dtype), distribution="sparse", seed=int(seed))


class ProjectionCache:
    """Memoise :class:`Projection` matrices by (shape, distribution, seed, device, dtype)."""

    def __init__(self) -> None:
        self.cache: dict[tuple[int, int, str, int, str, torch.dtype], Projection] = {}
        self.lock = threading.Lock()

    def get_or_build(
        self,
        out_dim: int,
        in_dim: int,
        *,
        distribution: Distribution,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype,
        projector: Projector | None = None,
    ) -> Projection:
        device_key = str(torch.device(device))
        key = (int(out_dim), int(in_dim), distribution, int(seed), device_key, dtype)
        with self.lock:
            existing = self.cache.get(key)
            if existing is not None:
                return existing
            if projector is None:
                if distribution == "gaussian":
                    projector = Gaussian()
                elif distribution == "rademacher":
                    projector = Rademacher()
                elif distribution == "sparse":
                    projector = Sparse()
                else:
                    raise ValueError(f"unknown distribution {distribution!r}")
            projection = projector.build(out_dim, in_dim, seed=seed, device=device, dtype=dtype)
            self.cache[key] = projection
            return projection

    def info(self) -> dict[str, int]:
        return {"size": len(self.cache), "max_size": len(self.cache)}

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()


CACHE: ProjectionCache = ProjectionCache()


def projection(
    out_dim: int,
    in_dim: int,
    *,
    distribution: Distribution = "gaussian",
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    projector: Projector | None = None,
) -> Projection:
    """Get-or-build a cached :class:`Projection`."""
    return CACHE.get_or_build(
        out_dim,
        in_dim,
        distribution=distribution,
        seed=seed,
        device=device,
        dtype=dtype,
        projector=projector,
    )


def projection_cache_info() -> dict[str, int]:
    return CACHE.info()


def clear_projection_cache() -> None:
    CACHE.clear()
