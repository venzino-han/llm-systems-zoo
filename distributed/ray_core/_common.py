"""Helpers shared by the Ray Core experiments (local-only, no harness changes)."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import ray

REPO_ROOT = Path(__file__).resolve().parents[2]


def init_ray(num_cpus: int = 8, num_gpus: float | None = None, **kwargs: Any) -> dict[str, Any]:
    """Start a local Ray cluster with a bounded footprint and return its resources."""
    if ray.is_initialized():
        ray.shutdown()
    opts: dict[str, Any] = dict(
        num_cpus=num_cpus,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
        runtime_env={"working_dir": None},
    )
    if num_gpus is not None:
        opts["num_gpus"] = num_gpus
    opts.update(kwargs)
    ray.init(**opts)
    return ray.cluster_resources()


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def nvidia_smi_used_mib() -> list[int]:
    """Used memory per visible GPU according to nvidia-smi (0 entries on CPU hosts)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return [int(x.strip()) for x in out.splitlines() if x.strip()]
    except Exception:
        return []


def spilled_mib() -> float | None:
    """Parse the amount spilled to disk from Ray's memory summary (None if unavailable)."""
    try:
        from ray._private.internal_api import memory_summary

        text = memory_summary(stats_only=True)
    except Exception:
        return None
    m = re.search(r"Spilled (\d+) MiB", text)
    return float(m.group(1)) if m else 0.0


def busy_wait(seconds: float) -> int:
    """CPU-bound work of a controllable duration (used to vary task granularity)."""
    t_end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < t_end:
        n += 1
    return n


def env_summary() -> dict[str, Any]:
    return {
        "ray": ray.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda": cuda_available(),
    }
