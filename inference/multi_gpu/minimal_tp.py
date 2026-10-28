"""Educational tensor parallelism: Megatron-style column/row parallel linears in ~100 lines.

A transformer block under TP splits every big matmul across `world` ranks:

    attention: QKV projection is *column* parallel (each rank owns a slice of heads),
               the output projection is *row* parallel and ends with an all-reduce.
    MLP:       up/gate projection is *column* parallel, down projection is *row* parallel
               and ends with an all-reduce.

So each block costs exactly two all-reduces of `[tokens, hidden]`, regardless of model
size, which is why TP across slow links hurts decode (tiny messages, latency bound) much
more than prefill (large messages, bandwidth bound, and compute dominates anyway).

This module verifies the sharded computation equals the dense one and times compute vs
communication. It runs with the gloo backend on CPU (tests) or NCCL on GPUs:

    python -m inference.multi_gpu.minimal_tp --spawn 2 --backend gloo
    torchrun --nproc_per_node=2 -m inference.multi_gpu.minimal_tp --backend nccl
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F


def shard(t: torch.Tensor, rank: int, world: int, dim: int) -> torch.Tensor:
    return t.chunk(world, dim=dim)[rank].contiguous()


class ColumnParallelLinear:
    """y = x @ W^T with W split along the output dimension: each rank produces a slice of y."""

    def __init__(self, weight: torch.Tensor, rank: int, world: int):
        self.w = shard(weight, rank, world, dim=0)  # [out/world, in]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.T  # [tokens, out/world]; no communication


class RowParallelLinear:
    """y = x @ W^T with W split along the input dimension: partial sums are all-reduced."""

    def __init__(self, weight: torch.Tensor, rank: int, world: int):
        self.w = shard(weight, rank, world, dim=1)  # [out, in/world]

    def __call__(self, x_shard: torch.Tensor, timer: dict[str, float] | None = None) -> torch.Tensor:
        y = x_shard @ self.w.T  # partial [tokens, out]
        t0 = time.perf_counter()
        dist.all_reduce(y)
        if timer is not None:
            timer["comm_s"] = timer.get("comm_s", 0.0) + time.perf_counter() - t0
        return y


class TPBlock:
    """Attention-projection + MLP sub-block (no softmax attention; the projections are what TP shards)."""

    def __init__(self, params: dict[str, torch.Tensor], rank: int, world: int):
        self.qkv = ColumnParallelLinear(params["w_qkv"], rank, world)
        self.o = RowParallelLinear(params["w_o"], rank, world)
        self.up = ColumnParallelLinear(params["w_up"], rank, world)
        self.down = RowParallelLinear(params["w_down"], rank, world)

    def __call__(self, x: torch.Tensor, timer: dict[str, float] | None = None) -> torch.Tensor:
        h = self.qkv(x)                 # local heads
        h = F.gelu(h)                   # stands in for attention over the local heads
        x = x + self.o(h, timer)        # all-reduce #1
        m = F.gelu(self.up(x))
        x = x + self.down(m, timer)     # all-reduce #2
        return x


def dense_block(x: torch.Tensor, p: dict[str, torch.Tensor]) -> torch.Tensor:
    h = F.gelu(x @ p["w_qkv"].T)
    x = x + h @ p["w_o"].T
    m = F.gelu(x @ p["w_up"].T)
    return x + m @ p["w_down"].T


def make_params(hidden: int, ffn: int, seed: int = 0, device: torch.device | None = None) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    scale = hidden**-0.5
    p = {
        "w_qkv": torch.randn(hidden, hidden, generator=g) * scale,
        "w_o": torch.randn(hidden, hidden, generator=g) * scale,
        "w_up": torch.randn(ffn, hidden, generator=g) * scale,
        "w_down": torch.randn(hidden, ffn, generator=g) * scale,
    }
    return {k: v.to(device) for k, v in p.items()} if device else p


def check_equivalence(rank: int, world: int, hidden: int, ffn: int, tokens: int, device: torch.device, dtype: torch.dtype) -> float:
    params = {k: v.to(dtype) for k, v in make_params(hidden, ffn, device=device).items()}
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(tokens, hidden, generator=g) * 0.1).to(device=device, dtype=dtype)
    y_tp = TPBlock(params, rank, world)(x)
    y_ref = dense_block(x, params)
    return (y_tp - y_ref).abs().max().item()


def time_block(rank: int, world: int, hidden: int, ffn: int, tokens: int, iters: int, device: torch.device, dtype: torch.dtype) -> dict[str, float]:
    params = {k: v.to(dtype) for k, v in make_params(hidden, ffn, device=device).items()}
    block = TPBlock(params, rank, world)
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    for _ in range(3):
        block(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dist.barrier()
    timer: dict[str, float] = {}
    t0 = time.perf_counter()
    for _ in range(iters):
        block(x, timer)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total = time.perf_counter() - t0
    comm = timer.get("comm_s", 0.0)
    return {"tokens": tokens, "total_ms_per_block": total / iters * 1e3, "comm_ms_per_block": comm / iters * 1e3,
            "comm_share": comm / total if total else 0.0, "all_reduce_bytes": tokens * hidden * x.element_size()}


def _worker(rank: int, world: int, backend: str, args: argparse.Namespace, init_method: str | None, result_file: str | None) -> None:
    if init_method:
        dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=world)
    else:
        dist.init_process_group(backend)
        rank, world = dist.get_rank(), dist.get_world_size()
    if backend == "nccl":
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    dtype = getattr(torch, args.dtype)
    err = check_equivalence(rank, world, args.hidden, args.ffn, args.tokens, device, dtype)
    rows = [time_block(rank, world, args.hidden, args.ffn, t, args.iters, device, dtype) for t in (1, 8, 64, args.tokens)]
    if rank == 0:
        tol = 1e-4 if dtype == torch.float32 else 5e-2
        print(f"backend={backend} world={world} hidden={args.hidden} ffn={args.ffn} dtype={args.dtype}")
        print(f"max |TP - dense| = {err:.3e}  ({'OK' if err < tol else 'MISMATCH'} at tol {tol})")
        print(f"{'tokens':>7s} {'block ms':>9s} {'comm ms':>8s} {'comm %':>7s} {'all-reduce bytes':>17s}")
        for r in rows:
            print(f"{r['tokens']:>7d} {r['total_ms_per_block']:>9.3f} {r['comm_ms_per_block']:>8.3f} {r['comm_share']*100:>6.1f}% {r['all_reduce_bytes']:>17d}")
        if result_file:
            import json

            with open(result_file, "w") as f:
                json.dump({"max_abs_err": err, "rows": rows, "world": world, "backend": backend}, f)
    dist.barrier()
    dist.destroy_process_group()


def run_spawned(world: int, backend: str = "gloo", **kw: Any) -> dict[str, Any]:
    """Helper for tests: spawn `world` processes, return rank-0 results."""
    import json

    import torch.multiprocessing as mp

    args = build_parser().parse_args([])
    for k, v in kw.items():
        setattr(args, k, v)
    tmp = tempfile.mkdtemp()
    init = f"file://{tmp}/init"
    result = f"{tmp}/result.json"
    mp.spawn(_worker, args=(world, backend, args, init, result), nprocs=world, join=True)
    with open(result) as f:
        return json.load(f)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--spawn", type=int, default=0, help="spawn N local processes (instead of torchrun)")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--ffn", type=int, default=4096)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--dtype", default="float32")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.spawn:
        import torch.multiprocessing as mp

        init = f"file://{tempfile.mkdtemp()}/init"
        mp.spawn(_worker, args=(args.spawn, args.backend, args, init, None), nprocs=args.spawn, join=True)
    else:
        _worker(0, 0, args.backend, args, None, None)


if __name__ == "__main__":
    main()
