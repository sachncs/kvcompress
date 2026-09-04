"""Decomposition strategies (exact and randomised SVD).

The :class:`Decomposer` ABC defines the contract; :class:`Exact` and
:class:`Randomized` are the two strategies. Use :class:`DecomposerRegistry`
to construct one by name.

Both strategies return a :class:`Decomposition` carrying the singular
triples plus a ``tail_mass`` scalar. The randomised path follows
Halko, Martinsson, Tropp (2011): Stage A finds an orthonormal basis for
the range via a Gaussian sketch plus power iterations; Stage B does an
exact small SVD on the projected matrix and lifts back to the original
space.

Thread-safety: instances are stateless across calls. Random components
use a per-call :class:`torch.Generator` instead of mutating the global
``torch.manual_seed`` state (which was Tier 0 bug B-07).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Type

import torch

from kvfold.errors import ShapeError

__all__ = ["Decomposition", "Decomposer", "Exact", "Randomized", "DecomposerRegistry"]


log = logging.getLogger(__name__)


@dataclass
class Decomposition:
    """Result of an SVD-style decomposition ``A ≈ U @ diag(S) @ Vh``.

    Attributes:
        u: shape ``(m, k)`` with orthonormal columns.
        s: shape ``(k,)`` non-negative singular values, descending.
        vh: shape ``(k, n)`` — right singular vectors, transposed.
        tail_mass: relative Frobenius mass discarded by the rank-``r``
            truncation. ``0.0`` when the full spectrum is retained.
        strategy: name of the strategy that produced this decomposition.
        full_s: full singular spectrum if it was computed (only :class:`Exact`
            always returns it). ``None`` for :class:`Randomized`.
    """

    u: torch.Tensor
    s: torch.Tensor
    vh: torch.Tensor
    tail_mass: float
    strategy: Literal["exact", "randomised"]
    full_s: torch.Tensor | None = None

    @property
    def rank(self) -> int:
        """Rank of the approximation — ``s.numel()``."""
        return int(self.s.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        """``(m, n)`` of the original matrix ``A``."""
        return (int(self.u.shape[0]), int(self.vh.shape[1]))

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct ``A ≈ U @ diag(S) · Vh``."""
        return (self.u * self.s) @ self.vh

    def reconstruction_error(self, a: torch.Tensor) -> float:
        """Relative Frobenius error ``||A - Â||F / ||A||F``.

        Returns ``0.0`` when ``||A||F == 0`` (degenerate case).
        """
        recon = self.reconstruct()
        num = torch.linalg.norm(a - recon)
        den = torch.linalg.norm(a)
        if float(den) == 0.0:
            return 0.0
        return float(num / den)


def tail_mass(s: torch.Tensor, r: int) -> float:
    """Relative Frobenius tail mass past the top-``r`` singular values."""
    if r >= s.shape[0]:
        return 0.0
    total = float(torch.sum(s * s))
    if total <= 0:
        return 0.0
    tail = float(torch.sum(s[r:] * s[r:]))
    return tail / total


class Decomposer(ABC):
    """Abstract strategy for low-rank decomposition."""

    name: Literal["exact", "randomised"] = "exact"

    @abstractmethod
    def decompose(self, a: torch.Tensor, rank: int) -> Decomposition:
        """Compute a rank-``rank`` decomposition of ``a``."""


class Exact(Decomposer):
    """Full-spectrum SVD; output truncated to ``rank`` if needed."""

    name: Literal["exact", "randomised"] = "exact"

    def decompose(self, a: torch.Tensor, rank: int) -> Decomposition:
        if a.dim() != 2:
            raise ShapeError(f"Exact decomposer expects 2-D matrix; got {a.dim()}-D shape {tuple(a.shape)}")
        compute_dtype = a.dtype if a.dtype in (torch.float32, torch.float64) else torch.float32
        a_compute = a.to(compute_dtype) if a.dtype != compute_dtype else a
        m, n = a.shape
        max_rank = min(m, n)
        rank = max(1, min(int(rank), max_rank))
        full = torch.linalg.svd(a_compute, full_matrices=False)
        u, s, vh = full.U, full.S, full.Vh
        tail = tail_mass(s, rank)
        u_out = u[:, :rank].contiguous()
        s_out = s[:rank].contiguous()
        vh_out = vh[:rank, :].contiguous()
        if a.dtype != compute_dtype:
            u_out = u_out.to(a.dtype)
            s_out = s_out.to(a.dtype)
            vh_out = vh_out.to(a.dtype)
        return Decomposition(
            u=u_out,
            s=s_out,
            vh=vh_out,
            tail_mass=tail,
            strategy="exact",
            full_s=s,
        )


class Randomized(Decomposer):
    """Halko-Martinsson-Tropp randomised SVD.

    Uses a per-call :class:`torch.Generator` so the global RNG state is
    never mutated.
    """

    name: Literal["exact", "randomised"] = "randomised"

    def __init__(self, *, oversampling: int = 10, n_power: int = 2, seed: int = 0, cap: int | None = None) -> None:
        self.oversampling = int(oversampling)
        self.n_power = int(n_power)
        self.seed = int(seed)
        self.cap = int(cap) if cap is not None else None

    def decompose(self, a: torch.Tensor, rank: int) -> Decomposition:
        if a.dim() != 2:
            raise ShapeError(f"Randomized decomposer expects 2-D matrix; got {a.dim()}-D shape {tuple(a.shape)}")
        m, n = a.shape
        max_rank = min(m, n)
        rank = max(1, min(int(rank), max_rank))
        k = rank + self.oversampling
        if self.cap is not None:
            k = min(k, self.cap)
        k = max(k, rank + 1)
        k = min(k, max_rank)

        compute_dtype = a.dtype if a.dtype in (torch.float32, torch.float64) else torch.float32
        a_compute = a.to(compute_dtype) if a.dtype != compute_dtype else a

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + rank)
        omega = torch.randn(n, k, dtype=compute_dtype, device=a_compute.device, generator=gen)
        y = a_compute @ omega

        for _ in range(self.n_power):
            y = a_compute @ (a_compute.t() @ y)

        q, _ = torch.linalg.qr(y, mode="reduced")
        b = q.t() @ a_compute
        u_b, s_b, vh_b = torch.linalg.svd(b, full_matrices=False)
        u = q @ u_b

        if rank > s_b.shape[0]:
            rank = s_b.shape[0]
        u_r = u[:, :rank].contiguous()
        s_r = s_b[:rank].contiguous()
        vh_r = vh_b[:rank, :].contiguous()

        a_fro_sq = float(torch.sum(a_compute * a_compute))
        retained = float(torch.sum(s_r * s_r))
        if a_fro_sq > 0:
            tail_mass_value = max(0.0, 1.0 - retained / a_fro_sq)
        else:
            tail_mass_value = 0.0

        if a.dtype != compute_dtype:
            u_r = u_r.to(a.dtype)
            s_r = s_r.to(a.dtype)
            vh_r = vh_r.to(a.dtype)

        log.debug(
            "Randomized: rank=%d oversampling=%d n_power=%d tail_mass=%.4e",
            rank,
            self.oversampling,
            self.n_power,
            tail_mass_value,
        )
        return Decomposition(
            u=u_r,
            s=s_r,
            vh=vh_r,
            tail_mass=tail_mass_value,
            strategy="randomised",
            full_s=None,
        )


class DecomposerRegistry:
    """Registry of :class:`Decomposer` strategies."""

    def __init__(self) -> None:
        self.entries: dict[str, Type[Decomposer]] = {
            "exact": Exact,
            "randomised": Randomized,
        }

    def register(self, name: str, cls: Type[Decomposer]) -> Type[Decomposer]:
        if not issubclass(cls, Decomposer):
            raise TypeError(f"{cls.__name__} must inherit from Decomposer")
        self.entries[name] = cls
        return cls

    def resolve(self, name: str) -> Type[Decomposer]:
        try:
            return self.entries[name]
        except KeyError:
            raise KeyError(f"unknown decomposer {name!r}; available: {list(self.entries)}") from None

    def default(self, method: Literal["auto", "exact", "randomised"], **kwargs: object) -> Decomposer:
        """Construct the appropriate strategy for ``method``.

        ``"auto"`` picks :class:`Exact` when no kwargs are given, else
        :class:`Randomized`.
        """
        if method == "auto":
            return Exact()
        if method == "exact":
            return Exact()
        if method == "randomised":
            return Randomized(**kwargs)  # type: ignore[arg-type]
        raise ValueError(f"unknown decomposer method {method!r}")


REGISTRY: DecomposerRegistry = DecomposerRegistry()


def tail_mass_at(s: torch.Tensor, r: int) -> float:
    """Public alias for :func:`tail_mass` for explicit semantics."""
    return tail_mass(s, r)
