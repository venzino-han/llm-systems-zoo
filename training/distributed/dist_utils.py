"""Process-group setup, scaling-mode batch arithmetic and communication probes shared by the
DDP and FSDP scripts."""

from __future__ import annotations

import os
import socket
from contextlib import closing
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: str
    backend: str


def free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def init_distributed(backend: str | None = None, rank: int | None = None, world_size: int | None = None) -> DistInfo:
    """Works under torchrun (env vars) or when called from a spawned test worker."""
    rank = int(os.environ.get("RANK", rank if rank is not None else 0))
    world_size = int(os.environ.get("WORLD_SIZE", world_size if world_size is not None else 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    use_cuda = torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", None) != ""
    backend = backend or ("nccl" if use_cuda else "gloo")
    device = f"cuda:{local_rank}" if backend == "nccl" else "cpu"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29571")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistInfo(rank, world_size, local_rank, device, backend)


def per_rank_batch(scaling: str, world_size: int, global_batch: int | None, per_rank: int | None) -> tuple[int, int]:
    """Return (per_rank_batch, global_batch).

    strong scaling: the total work (global batch) is fixed; each GPU gets global/world.
    weak scaling:   the work per GPU is fixed; the global batch grows with the world size.
    """
    if scaling == "strong":
        assert global_batch is not None and global_batch % world_size == 0, "global batch must divide by world size"
        return global_batch // world_size, global_batch
    assert per_rank is not None
    return per_rank, per_rank * world_size


def all_gather_object(obj: Any, world_size: int) -> list[Any]:
    out: list[Any] = [None] * world_size
    dist.all_gather_object(out, obj)
    return out


def comm_kernel_time(model, batches, step_fn, n_steps: int = 3) -> dict[str, float]:
    """Sum the device time of NCCL kernels over `n_steps` optimizer steps with the torch profiler.

    This is the *total* communication time, including the part that overlaps with compute.
    Comparing it with the exposed time from `exposed_comm_time` gives the overlap ratio.
    """
    if not torch.cuda.is_available():
        return {"comm_kernel_s_per_step": 0.0, "total_kernel_s_per_step": 0.0}
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(n_steps):
            step_fn(next(batches))
        torch.cuda.synchronize()
    comm = 0.0
    total = 0.0
    for ev in prof.key_averages():
        t = getattr(ev, "device_time_total", getattr(ev, "cuda_time_total", 0.0)) / 1e6
        if t <= 0:
            continue
        total += t
        name = ev.key.lower()
        if "nccl" in name or "allreduce" in name or "all_gather" in name or "reduce_scatter" in name or "c10d" in name:
            comm += t
    return {"comm_kernel_s_per_step": comm / n_steps, "total_kernel_s_per_step": total / n_steps}
