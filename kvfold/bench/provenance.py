"""Reproducibility metadata shared by benchmark entry points."""

from __future__ import annotations

import platform
import sys
from typing import Any

import torch


def environment_metadata(*, seed: int, warmup: int, iterations: int) -> dict[str, Any]:
    """Return runtime and measurement metadata for a benchmark artifact."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "seed": seed,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
    }


__all__ = ["environment_metadata"]
