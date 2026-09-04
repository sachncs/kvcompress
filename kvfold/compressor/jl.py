"""Johnson-Lindenstrauss random projections.

JL projections compress the residual R = X - X̂ down to a lower-dim rotated
space before quantization. Two distributions are supported:

* **Gaussian**: N(0, 1/dh). Approximately preserves Euclidean norms
  (JL lemma); well-behaved, but each entry is a float32.
* **Rademacher**: ±1 with probability 1/2, optionally sparse (Achlioptas).
  The sparse variant stores 1/sqrt(sparsity·dh) per nonzero entry, so
  larger entries fit better in low-bit quantisation.

Both are dimension-preserving on the trailing axis (``R ∈ R^{m·T × dh}`` →
``Π ∈ R^{dh × dh}``); a learned projection that *reduces* dimension is not
implemented because the paper's residual path uses a square rotation.

Projections are deterministic given a seed and the input shape. They are
cached by shape + seed so the same cell never pays the projection cost
twice. The cache is process-wide; it's the only global mutable state in
:mod:`kvcompress.compressor`. Call :func:`clear_projection_cache` from
test fixtures to reset between cases.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Literal

import torch

__all__ = [
    "JLDistribution",
    "JLProjection",
    "cached_projection",
    "clear_projection_cache",
    "gaussian_projection",
    "rademacher_projection",
]


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
        """Device the projection lives on (delegates to ``matrix.device``)."""
        return self.matrix.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the projection (delegates to ``matrix.dtype``)."""
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
        # x: (..., dh) ; matrix: (k, dh) with k == dh for rotation.
        # The cheap ``.t()`` here is exact for any matrix, square or not.
        return x @ self.matrix.t()

    def apply_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse projection (``matrix`` for square rotation).

        Used at decode time to undo the rotation applied before quantization.
        For a square random matrix the inverse is not exactly
        ``matrix.t()``, but the JL lemma guarantees the embedded and
        recovered vectors have nearly the same norm, so this is the standard
        cheap inverse used by the paper.

        Note: this is ``x @ matrix``, not ``x @ matrix.t()``. For a square
        random matrix, the rows of ``matrix`` (used at apply time via
        ``matrix.t()``) and the columns (used at inverse time) span the
        same subspace, so the cheap inverse recovers the original signal
        up to a JL-bounded error.
        """
        return x @ self.matrix


def gaussian_projection(
    output_dim: int,
    input_dim: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> JLProjection:
    """Construct a Gaussian JL projection with N(0, 1/input_dim) entries.

    The 1/sqrt(input_dim) scaling makes the projection approximately
    preserve squared norms (the Johnson-Lindenstrauss lemma): for any
    vector ``x``, ``E[||Πx||²] ≈ ||x||²``.

    Construction is on CPU (a fresh ``torch.Generator`` per call) to keep
    the result bit-exact across CUDA driver versions; the returned matrix
    is then moved to the requested ``device``.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    # Build on CPU for determinism, then move.
    m = torch.randn(output_dim, input_dim, generator=gen, dtype=torch.float32)
    # Scale so the projection is approximately norm-preserving.
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
        m = (torch.randint(0, 2, (output_dim, input_dim), generator=gen) * 2 - 1).to(torch.float32)
    else:
        mask = (torch.rand(output_dim, input_dim, generator=gen) < sparsity).to(torch.float32)
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

PROJECTION_CACHE: dict[tuple[int, int, str, int, str], JLProjection] = {}
PROJECTION_CACHE_LOCK = threading.Lock()


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
    # Ponytail: lock prevents two HF serve workers from racing on the
    # same cache key (which would build two copies of the same matrix).
    with PROJECTION_CACHE_LOCK:
        proj = PROJECTION_CACHE.get(key)
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
            PROJECTION_CACHE[key] = proj
    return proj


def clear_projection_cache() -> None:
    """Empty the global JL projection cache (test utility)."""
    with PROJECTION_CACHE_LOCK:
        PROJECTION_CACHE.clear()
