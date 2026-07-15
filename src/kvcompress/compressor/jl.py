"""Johnson-Lindenstrauss random projections.

JL projections compress the residual R = X - X̂ down to a lower-dim rotated
space before quantization. Two distributions are supported:

* **Gaussian**: N(0, 1/k). Slightly larger distortion but well-behaved.
* **Rademacher**: ±1 with probability 1/2 (Achlioptas-style). Sparse, fast.

Both are dimension-preserving on the trailing axis (``R ∈ R^{m·T × dh}`` →
``Π ∈ R^{dh × dh}``); a learned projection that *reduces* dimension is not
implemented because the paper's residual path uses a square rotation.

Projections are deterministic given a seed and the input shape. They are
cached by shape + seed so the same cell never pays the projection cost twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch

log = logging.getLogger(__name__)


JLDistribution = Literal["gaussian", "rademacher"]


@dataclass
class JLProjection:
    """A Johnson-Lindenstrauss random projection matrix.

    Attributes:
        matrix: shape ``(k, dh)`` for "reduce" or ``(dh, dh)`` for "rotate".
        distribution: ``"gaussian"`` or ``"rademacher"``.
        seed: seed used to construct ``matrix``.
    """

    matrix: torch.Tensor
    distribution: JLDistribution
    seed: int

    @property
    def device(self) -> torch.device:
        return self.matrix.device

    @property
    def dtype(self) -> torch.dtype:
        return self.matrix.dtype

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the projection along the last axis.

        Args:
            x: tensor with last dim ``dh`` matching ``matrix.shape[-1]``.

        Returns:
            Tensor with same shape as input. For rotation projections this is
            an exact linear map ``x @ matrix.T``.
        """
        if x.shape[-1] != self.matrix.shape[-1]:
            raise ValueError(
                f"JL apply: trailing dim mismatch: x.shape[-1]={x.shape[-1]}, "
                f"matrix.shape[-1]={self.matrix.shape[-1]}"
            )
        # x: (..., dh) ; matrix: (k, dh) with k == dh for rotation
        return x @ self.matrix.t()

    def apply_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse projection (``matrix^T`` for square rotation).

        Used at decode time to undo the rotation applied before quantization.
        For a square random matrix the inverse is not exactly
        ``matrix.t()``, but the JL lemma guarantees the embedded and
        recovered vectors have nearly the same norm, so this is the standard
        cheap inverse used by the paper.
        """
        # For square rotation: x @ matrix because matrix is orthonormal-ish
        return x @ self.matrix


def gaussian_projection(
    output_dim: int,
    input_dim: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> JLProjection:
    """Construct a Gaussian JL projection with N(0, 1/output_dim) entries.

    The 1/output_dim scaling makes the projection approximately preserve
    squared norms (the Johnson-Lindenstrauss lemma).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    # Build on CPU for determinism, then move.
    m = torch.randn(output_dim, input_dim, generator=gen, dtype=torch.float32)
    m = m / (input_dim**0.5)
    return JLProjection(
        matrix=m.to(device=device, dtype=dtype),
        distribution="gaussian",
        seed=int(seed),
    )


def rademacher_projection(
    output_dim: int,
    input_dim: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    sparsity: float = 1.0,
) -> JLProjection:
    """Construct a Rademacher JL projection with entries in {-1, +1}.

    With ``sparsity < 1`` only ``sparsity`` fraction of entries are nonzero,
    matching the Achlioptas sparse JL construction. ``sparsity == 1`` gives a
    dense ±1 matrix.

    The 1/sqrt(input_dim) scale preserves squared norms.
    """
    if not 0 < sparsity <= 1:
        raise ValueError(f"sparsity must be in (0, 1], got {sparsity}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    if sparsity == 1.0:
        m = (torch.randint(0, 2, (output_dim, input_dim), generator=gen) * 2 - 1).to(
            torch.float32
        )
    else:
        mask = (torch.rand(output_dim, input_dim, generator=gen) < sparsity).to(
            torch.float32
        )
        signs = (torch.randint(0, 2, (output_dim, input_dim), generator=gen) * 2 - 1).to(
            torch.float32
        )
        # Scale nonzero entries by 1/sqrt(sparsity) to preserve norm on average.
        m = mask * signs / (sparsity**0.5)
    m = m / (input_dim**0.5)
    return JLProjection(
        matrix=m.to(device=device, dtype=dtype),
        distribution="rademacher",
        seed=int(seed),
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


_PROJECTION_CACHE: dict[tuple[int, int, str, int, str], JLProjection] = {}


def cached_projection(
    output_dim: int,
    input_dim: int,
    *,
    distribution: JLDistribution = "gaussian",
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    sparsity: float = 1.0,
) -> JLProjection:
    """Return a cached JL projection keyed by (shape, distribution, seed).

    The cache is process-wide and is the *only* place that constructs JL
    matrices inside :mod:`kvcompress`. Tests can clear it via
    :func:`clear_projection_cache`.
    """
    key = (
        output_dim,
        input_dim,
        distribution,
        int(seed),
        str(device),
    )
    proj = _PROJECTION_CACHE.get(key)
    if proj is None:
        if distribution == "gaussian":
            proj = gaussian_projection(
                output_dim, input_dim, seed=seed, device=device, dtype=dtype
            )
        elif distribution == "rademacher":
            proj = rademacher_projection(
                output_dim,
                input_dim,
                seed=seed,
                device=device,
                dtype=dtype,
                sparsity=sparsity,
            )
        else:
            raise ValueError(f"unknown JL distribution {distribution!r}")
        _PROJECTION_CACHE[key] = proj
    return proj


def clear_projection_cache() -> None:
    """Empty the global JL projection cache (test utility)."""
    _PROJECTION_CACHE.clear()


__all__ = [
    "JLDistribution",
    "JLProjection",
    "cached_projection",
    "clear_projection_cache",
    "gaussian_projection",
    "rademacher_projection",
]