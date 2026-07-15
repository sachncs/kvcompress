"""Compression profiler — instrument compress / decompress calls.

A tiny per-call recorder that captures timing and size statistics. Useful
for the benchmark suite and for inspecting a real generation's KV cache
behaviour.

The profiler is *passive*: it doesn't change the algorithm's runtime
behaviour beyond the per-call wall-clock measurement. Disabling it via
the private ``_enabled`` flag removes the cost entirely.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class _CallRecord:
    """One profiled call."""

    name: str
    duration_ms: float
    bytes_in: int = 0
    bytes_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
        }


@dataclass
class CompressionProfiler:
    """Records timings of named calls."""

    records: list[_CallRecord] = field(default_factory=list)
    _enabled: bool = True

    @contextmanager
    def record(self, name: str, *, bytes_in: int = 0, bytes_out: int = 0):
        """Time a code block."""
        if not self._enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.records.append(
                _CallRecord(
                    name=name, duration_ms=elapsed_ms, bytes_in=bytes_in, bytes_out=bytes_out
                )
            )

    def summary(self) -> dict[str, dict[str, Any]]:
        """Aggregate records by name."""
        out: dict[str, dict[str, Any]] = {}
        for r in self.records:
            agg = out.setdefault(
                r.name, {"count": 0, "total_ms": 0.0, "bytes_in": 0, "bytes_out": 0}
            )
            agg["count"] += 1
            agg["total_ms"] += r.duration_ms
            agg["bytes_in"] += r.bytes_in
            agg["bytes_out"] += r.bytes_out
        for name, agg in out.items():
            agg["mean_ms"] = agg["total_ms"] / agg["count"]
        return out

    def reset(self) -> None:
        self.records.clear()


__all__ = ["CompressionProfiler"]
