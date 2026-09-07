"""Reproducibility helpers for all random components in kvfold.

The :class:`Seed` context manager restores the global torch RNG state on
exit, so a randomised operation (SVD sketch, JL projection matrix
construction) never leaks randomness into other code paths.

The :func:`generator` helper returns a per-device :class:`torch.Generator`
seeded with ``seed``. Concrete consumers (``Svd``, ``JLProj``) use per-call
generators instead of mutating the global seed, which fixes the latent
thread-safety hazard in the previous SVD implementation.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch


class Seed:
    """Context manager that pins the torch RNG state during its body.

    On ``__enter__`` the current CPU and CUDA RNG states are captured.
    On ``__exit__`` the captured states are restored, so any RNG mutation
    performed inside the ``with`` block is invisible to surrounding code.

    Example::

        with Seed(0):
            a = torch.randn(4)        # deterministic given seed 0
        b = torch.randn(4)            # original RNG state, not seeded
    """

    def __init__(self, value: int = 0) -> None:
        self.value = int(value)
        self.saved_cpu: torch.Tensor | None = None
        self.saved_cuda: dict[torch.device, torch.Tensor] = {}

    def __enter__(self) -> "Seed":
        self.saved_cpu = torch.get_rng_state()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f"cuda:{i}")
                self.saved_cuda[device] = torch.cuda.get_rng_state(device)
        torch.manual_seed(self.value)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.saved_cpu is not None:
            torch.set_rng_state(self.saved_cpu)
        for device, state in self.saved_cuda.items():
            torch.cuda.set_rng_state(state, device)

    def __repr__(self) -> str:
        return f"Seed(value={self.value})"


@contextlib.contextmanager
def generator(device: torch.device | str = "cpu", seed: int = 0) -> Iterator[torch.Generator]:
    """Yield a per-call :class:`torch.Generator` seeded with ``seed``.

    The generator is constructed fresh on entry and discarded on exit.
    CPU generators are always supported; CUDA generators are used when
    ``device`` is a CUDA device and CUDA is available. Otherwise a CPU
    generator is returned and the caller is expected to handle the device
    transfer explicitly.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    yield g


__all__ = ["Seed", "generator"]
