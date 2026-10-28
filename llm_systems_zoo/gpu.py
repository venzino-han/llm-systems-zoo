"""GPU helpers that do not require torch: free-memory polling for shared machines."""

from __future__ import annotations

import shutil
import subprocess
import time


def free_memory_gib(device: int = 0) -> float | None:
    """Free memory on a physical GPU index as nvidia-smi reports it (None if unavailable)."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={device}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10, stderr=subprocess.DEVNULL,
        ).strip()
        return int(out.splitlines()[0]) / 1024
    except Exception:
        return None


def wait_for_free_memory(need_gib: float, device: int = 0, timeout_s: float = 300, poll_s: float = 5) -> float:
    """Block until `need_gib` is free on `device` or `timeout_s` elapses.

    Returns the free memory observed last. Raises TimeoutError if the budget never became available,
    so a lab script fails loudly instead of OOM-ing another tenant's job on a shared GPU.
    """
    deadline = time.monotonic() + timeout_s
    free = free_memory_gib(device)
    while free is not None and free < need_gib:
        if time.monotonic() > deadline:
            raise TimeoutError(f"GPU {device}: needed {need_gib:.1f} GiB free, saw {free:.1f} GiB after {timeout_s}s")
        time.sleep(poll_s)
        free = free_memory_gib(device)
    return free if free is not None else float("nan")
