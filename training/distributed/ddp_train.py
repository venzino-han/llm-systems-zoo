"""L05: DistributedDataParallel training with throughput, memory and communication accounting.

    torchrun --nproc_per_node 2 training/distributed/ddp_train.py --model small --dataset smoltalk \
        --scaling strong --global-batch 8 --steps 30

    # single process baseline for the scaling table (same script, world size 1)
    torchrun --nproc_per_node 1 training/distributed/ddp_train.py --model small --dataset smoltalk \
        --scaling strong --global-batch 8 --steps 30

DDP keeps a full replica of the model, gradients and optimizer state on every rank and
all-reduces gradients bucket by bucket during backward, overlapping communication with the
remaining backward compute. Memory per rank therefore does not shrink with more GPUs; only the
batch does (strong scaling) or the work grows (weak scaling).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from llm_systems_zoo import PhaseTimer, Run
from training.common import (
    TINY_RANDOM,
    TrainConfig,
    build_model,
    build_tokenizer,
    iter_batches,
    load_samples,
    make_optimizer,
    optimizer_step,
    resolve_model_id,
    train_loop,
)
from training.distributed.dist_utils import (
    all_gather_object,
    comm_kernel_time,
    init_distributed,
    per_rank_batch,
)

LAB = "L05_training_efficiency"


def exposed_comm_time(model: DDP, batches, cfg: TrainConfig, n_steps: int = 5) -> dict[str, float]:
    """Exposed (non-overlapped) all-reduce cost: fwd+bwd time with gradient sync minus fwd+bwd
    time under `no_sync()`. Optimizer step excluded on both sides."""
    timer = PhaseTimer(sync_cuda=str(cfg.device).startswith("cuda"))
    opt = make_optimizer(model, 0.0)  # lr 0: the probe must not change the weights

    def fwd_bwd(sync: bool):
        b = next(batches)
        ctx = model.no_sync() if not sync else torch.autograd.grad_mode.enable_grad()
        with ctx, timer.phase("sync" if sync else "nosync"):
            optimizer_step(model, opt, [b], cfg, PhaseTimer())
        opt.zero_grad(set_to_none=True)

    for _ in range(2):  # warm both paths
        fwd_bwd(True)
        fwd_bwd(False)
    timer.reset()
    for _ in range(n_steps):
        fwd_bwd(True)
        fwd_bwd(False)
    s = timer.summary()
    exposed = s["sync"]["mean_s"] - s["nosync"]["mean_s"]
    return {"step_with_sync_s": s["sync"]["mean_s"], "step_no_sync_s": s["nosync"]["mean_s"],
            "exposed_comm_s": max(exposed, 0.0)}


def run(cfg: TrainConfig, args: argparse.Namespace, *, backend: str | None = None, rank: int | None = None,
        world_size: int | None = None, runs_root: str = "runs") -> dict[str, Any] | None:
    info = init_distributed(backend, rank, world_size)
    cfg.device = info.device
    torch.manual_seed(cfg.seed)
    pr_batch, global_batch = per_rank_batch(args.scaling, info.world_size, args.global_batch, args.per_rank_batch)
    cfg.batch_size = pr_batch
    cfg.seed = cfg.seed + 1000 * info.rank  # different data per rank

    model_id = resolve_model_id(cfg.model)
    tok = None if cfg.model == TINY_RANDOM else build_tokenizer(model_id)
    pad_id = 0 if tok is None else tok.pad_token_id
    samples = load_samples(cfg, tok)
    model, minfo = build_model(cfg)
    ddp = DDP(model, device_ids=[info.local_rank] if info.backend == "nccl" else None,
              bucket_cap_mb=args.bucket_cap_mb, gradient_as_bucket_view=True)

    t0 = time.perf_counter()
    summary = train_loop(ddp, cfg, samples, pad_id, no_sync_ctx=ddp.no_sync)
    summary["wall_s"] = time.perf_counter() - t0
    probes: dict[str, float] = {}
    if args.probe_comm:
        probes.update(exposed_comm_time(ddp, iter_batches(samples, cfg, pad_id), cfg))
        if torch.cuda.is_available():
            opt = make_optimizer(ddp, 0.0)
            probes.update(comm_kernel_time(ddp, iter_batches(samples, cfg, pad_id),
                                           lambda b: optimizer_step(ddp, opt, [b], cfg, PhaseTimer())))
    local = {"rank": info.rank, "real_tokens_per_s": summary["real_tokens_per_s"], "mean_step_s": summary["mean_step_s"],
             "peak_allocated_gib": summary["peak_allocated_gib"], "peak_reserved_gib": summary["peak_reserved_gib"],
             "final_loss": summary["final_loss"], **probes}
    gathered = all_gather_object(local, info.world_size)
    result = None
    if info.rank == 0:
        agg_tps = sum(g["real_tokens_per_s"] for g in gathered)
        result = {
            "strategy": "ddp", "scaling": args.scaling, "world_size": info.world_size, "backend": info.backend,
            "global_batch": global_batch, "per_rank_batch": pr_batch, "grad_accum": cfg.grad_accum,
            "aggregate_real_tokens_per_s": agg_tps, "per_rank": gathered,
            "mean_step_s": max(g["mean_step_s"] for g in gathered),  # the slowest rank bounds the step
            "peak_allocated_gib_max": max(g["peak_allocated_gib"] for g in gathered),
            "model_info": minfo, "config": cfg.to_dict(), "losses": summary["losses"], "phase_share": summary["phase_share"],
        }
        if runs_root is not None:
            with Run(LAB, f"ddp_w{info.world_size}_{args.scaling}", {**cfg.to_dict(), "scaling": args.scaling, "world_size": info.world_size},
                     root=runs_root, model={"id": model_id, "parallelism": f"DDP x{info.world_size}"},
                     workload={"dataset": cfg.dataset, "global_batch": global_batch, "per_rank_batch": pr_batch, "seq_len": cfg.seq_len},
                     execution={"steps": cfg.steps, "warmup_steps": cfg.warmup_steps, "bucket_cap_mb": args.bucket_cap_mb}) as r:
                r.update(model=minfo)
                r.save_summary(result)
                result["run_dir"] = str(r.dir)
            print(f"[ddp w={info.world_size} {args.scaling}] aggregate {agg_tps:.0f} tok/s, slowest-rank step {result['mean_step_s']*1000:.1f} ms, "
                  f"peak alloc {result['peak_allocated_gib_max']:.2f} GiB, probes {probes}")
    dist.barrier()
    dist.destroy_process_group()
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="small")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "smoltalk", "wikitext"])
    p.add_argument("--precision", default="bf16_autocast", choices=["fp32", "bf16_autocast", "fp16_autocast", "bf16_weights"])
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--packing", action="store_true")
    p.add_argument("--lora", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-samples", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scaling", choices=["strong", "weak"], default="strong")
    p.add_argument("--global-batch", type=int, default=8, help="strong scaling: fixed total batch")
    p.add_argument("--per-rank-batch", type=int, default=4, help="weak scaling: fixed batch per GPU")
    p.add_argument("--bucket-cap-mb", type=int, default=25)
    p.add_argument("--probe-comm", action="store_true", help="measure exposed and total communication time")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--runs-root", default="runs")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cfg = TrainConfig(model=a.model, dataset=a.dataset, precision=a.precision, grad_checkpointing=a.grad_checkpointing,
                      grad_accum=a.grad_accum, packing=a.packing, lora=a.lora, seq_len=a.seq_len, steps=a.steps,
                      warmup_steps=a.warmup_steps, lr=a.lr, n_samples=a.n_samples, seed=a.seed)
    if a.smoke:
        cfg.steps, cfg.warmup_steps, cfg.seq_len, cfg.n_samples = 6, 2, 128, 64
    run(cfg, a, runs_root=a.runs_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
