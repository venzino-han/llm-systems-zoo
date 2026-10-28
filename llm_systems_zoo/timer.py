"""Phase timers. Use them to decompose a pipeline (rollout / reward / learner, prefill / decode,
tool wait / model compute) instead of reporting one opaque end-to-end number."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator


def cuda_sync() -> None:
    """Synchronise the current CUDA device if torch+CUDA are present; otherwise a no-op."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class PhaseTimer:
    """Accumulates wall-clock time per named phase.

    >>> t = PhaseTimer(sync_cuda=True)
    >>> with t.phase("rollout"):
    ...     generate()
    >>> t.summary()["rollout"]["total_s"]
    """

    def __init__(self, sync_cuda: bool = False) -> None:
        self.sync_cuda = sync_cuda
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._t_start = time.perf_counter()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if self.sync_cuda:
            cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda:
                cuda_sync()
            dt = time.perf_counter() - t0
            self.totals[name] += dt
            self.counts[name] += 1
            self.samples[name].append(dt)

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] += seconds
        self.counts[name] += 1
        self.samples[name].append(seconds)

    def elapsed(self) -> float:
        return time.perf_counter() - self._t_start

    def summary(self) -> dict[str, Any]:
        wall = self.elapsed()
        accounted = sum(self.totals.values())
        out: dict[str, Any] = {}
        for name, total in self.totals.items():
            n = self.counts[name]
            out[name] = {
                "total_s": total,
                "count": n,
                "mean_s": total / n if n else 0.0,
                "share_of_wall": total / wall if wall > 0 else 0.0,
            }
        out["_wall_s"] = wall
        out["_unaccounted_s"] = max(wall - accounted, 0.0)
        return out

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()
        self.samples.clear()
        self._t_start = time.perf_counter()
