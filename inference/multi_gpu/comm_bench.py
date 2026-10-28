"""Collective-communication micro-benchmark: what does one TP all-reduce cost on this host?

    torchrun --nproc_per_node=2 -m inference.multi_gpu.comm_bench --backend nccl
    python -m inference.multi_gpu.comm_bench --backend gloo --spawn 2      # CPU fallback

For every message size it reports the time per all_reduce / all_gather and the derived
bandwidths using the NCCL-tests convention:
    algbw = bytes / time
    busbw = algbw * 2 * (n - 1) / n     (ring all-reduce moves 2(n-1)/n bytes per rank)

Why it matters for TP inference: a decoder layer under tensor parallelism performs two
all-reduces per layer (after attention and after the MLP) on a tensor of
`batch_tokens × hidden × dtype_bytes`. During decode `batch_tokens` is the number of
sequences in the step, so the message is tiny (KBs) and *latency*, not bandwidth, dominates:
`layers × 2 × all_reduce_latency` is a floor on TPOT.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Sequence

import torch
import torch.distributed as dist

from llm_systems_zoo.io import dump_json

DEFAULT_SIZES_BYTES = [4 * 1024, 64 * 1024, 1024**2, 16 * 1024**2, 128 * 1024**2]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_collective(op: str, tensor: torch.Tensor, iters: int, warmup: int, device: torch.device) -> float:
    """Mean seconds per collective (includes a barrier-free launch; timing is per rank)."""
    world = dist.get_world_size()
    if op == "all_reduce":
        fn = lambda: dist.all_reduce(tensor)  # noqa: E731
    elif op == "all_gather":
        out = [torch.empty_like(tensor) for _ in range(world)]
        fn = lambda: dist.all_gather(out, tensor)  # noqa: E731
    else:
        raise ValueError(op)
    for _ in range(warmup):
        fn()
    _sync(device)
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    dt = (time.perf_counter() - t0) / iters
    return dt


def run_benchmark(sizes: Sequence[int], iters: int, warmup: int, dtype: torch.dtype, device: torch.device) -> list[dict[str, Any]]:
    world = dist.get_world_size()
    rows = []
    for nbytes in sizes:
        n = max(1, nbytes // torch.tensor([], dtype=dtype).element_size())
        x = torch.randn(n, dtype=dtype, device=device) if dtype.is_floating_point else torch.ones(n, dtype=dtype, device=device)
        for op in ("all_reduce", "all_gather"):
            dt = bench_collective(op, x, iters, warmup, device)
            algbw = nbytes / dt / 1e9
            busbw = algbw * 2 * (world - 1) / world if op == "all_reduce" else algbw * (world - 1) / world
            rows.append({"op": op, "bytes": nbytes, "mean_us": dt * 1e6, "algbw_gbps": algbw, "busbw_gbps": busbw, "world_size": world})
    return rows


def interpolate_us(rows: Sequence[dict[str, Any]], nbytes: int) -> float:
    """Linear interpolation of the measured mean_us at `nbytes` (clamped to the measured range)."""
    pts = sorted((r["bytes"], r["mean_us"]) for r in rows)
    if nbytes <= pts[0][0]:
        return pts[0][1]
    if nbytes >= pts[-1][0]:
        return pts[-1][1]
    for (b0, t0), (b1, t1) in zip(pts, pts[1:]):
        if b0 <= nbytes <= b1:
            return t0 + (nbytes - b0) / (b1 - b0) * (t1 - t0)
    return pts[-1][1]


def tp_decode_floor(rows: Sequence[dict[str, Any]], layers: int = 28, hidden: int = 3584, batch: int = 1, dtype_bytes: int = 2) -> dict[str, Any]:
    """Per-token communication floor for TP decode: 2 all-reduces per layer of `batch × hidden × dtype`
    bytes, at the (interpolated) measured all_reduce latency for that message size. Defaults: Qwen2.5-7B."""
    msg = batch * hidden * dtype_bytes
    ar = [r for r in rows if r["op"] == "all_reduce"]
    if not ar:
        return {}
    us = interpolate_us(ar, msg)
    per_token_us = 2 * layers * us
    return {"message_bytes": msg, "all_reduce_us": us, "all_reduces_per_token": 2 * layers, "comm_floor_per_token_ms": per_token_us / 1000,
            "assumes": f"{layers} layers, hidden {hidden}, batch {batch}, {dtype_bytes}-byte dtype; latency interpolated between measured sizes"}


def _worker_main(rank: int, world: int, backend: str, args: argparse.Namespace, init_method: str | None) -> None:
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
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else DEFAULT_SIZES_BYTES
    rows = run_benchmark(sizes, args.iters, args.warmup, dtype, device)
    if rank == 0:
        floor = tp_decode_floor(rows, args.layers, args.hidden, args.batch, torch.tensor([], dtype=dtype).element_size())
        print(f"backend={backend} world={world} device={device} dtype={args.dtype}")
        print(f"{'op':10s} {'bytes':>12s} {'mean_us':>10s} {'algbw GB/s':>11s} {'busbw GB/s':>11s}")
        for r in rows:
            print(f"{r['op']:10s} {r['bytes']:>12d} {r['mean_us']:>10.1f} {r['algbw_gbps']:>11.2f} {r['busbw_gbps']:>11.2f}")
        if floor:
            print(f"TP decode comm floor: {floor['comm_floor_per_token_ms']:.2f} ms/token ({floor['all_reduces_per_token']} all-reduces of ~{floor['message_bytes']} B at {floor['all_reduce_us']:.1f} us; {floor['assumes']})")
        if args.out:
            dump_json({"backend": backend, "world_size": world, "device": str(device), "dtype": args.dtype, "rows": rows, "tp_decode_floor": floor,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if backend == "nccl" else []}, args.out)
            print("wrote", args.out)
    dist.barrier()
    dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--spawn", type=int, default=0, help="spawn N local processes instead of torchrun (gloo/CPU tests)")
    p.add_argument("--sizes", default="", help="comma-separated message sizes in bytes")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--layers", type=int, default=28)
    p.add_argument("--hidden", type=int, default=3584)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--out", default=None, help="write JSON results here")
    p.add_argument("--smoke", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.sizes, args.iters, args.warmup = "4096,1048576", 10, 2
    if args.spawn:
        import tempfile

        import torch.multiprocessing as mp

        init = f"file://{tempfile.mkdtemp()}/init"
        mp.spawn(_worker_main, args=(args.spawn, args.backend, args, init), nprocs=args.spawn, join=True)
    else:
        _worker_main(0, 0, args.backend, args, None)


if __name__ == "__main__":
    main()
    _ = json
