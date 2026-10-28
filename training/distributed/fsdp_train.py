"""L05: FSDP2 (`fully_shard`) training with per-rank memory and communication accounting.

    torchrun --nproc_per_node 2 training/distributed/fsdp_train.py --model small --dataset smoltalk \
        --scaling strong --global-batch 8 --steps 30

FSDP shards parameters, gradients and optimizer state across ranks (ZeRO-3 style). Each layer's
parameters are all-gathered right before its forward/backward and freed afterwards; gradients are
reduce-scattered. Memory per rank falls roughly with the world size, at the price of more
communication volume than DDP (all-gather in forward *and* backward, plus reduce-scatter).
Falls back to FSDP1 (`FullyShardedDataParallel`) when `fully_shard` is unavailable.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from typing import Any

import torch
import torch.distributed as dist

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
from training.distributed.ddp_train import build_parser
from training.distributed.dist_utils import (
    all_gather_object,
    comm_kernel_time,
    init_distributed,
    per_rank_batch,
)

LAB = "L05_training_efficiency"


def _decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    for attr in ("model.layers", "transformer.h", "layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    return []


def wrap_fsdp(model: torch.nn.Module, cfg: TrainConfig, device: str, reshard_after_forward: bool = True) -> tuple[torch.nn.Module, str]:
    """Shard every decoder layer, then the root (embeddings + head). Returns (model, api_used)."""
    bf16 = cfg.precision.startswith("bf16")
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if bf16 else torch.float32, reduce_dtype=torch.float32)
        for layer in _decoder_layers(model):
            fully_shard(layer, mp_policy=mp, reshard_after_forward=reshard_after_forward)
        fully_shard(model, mp_policy=mp, reshard_after_forward=reshard_after_forward)
        return model, "fsdp2"
    except ImportError:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        layer_types = {type(m) for m in _decoder_layers(model)}
        policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_types)
        mp = MixedPrecision(param_dtype=torch.bfloat16 if bf16 else torch.float32, reduce_dtype=torch.float32)
        model = FSDP(model, auto_wrap_policy=policy, mixed_precision=mp, sharding_strategy=ShardingStrategy.FULL_SHARD,
                     device_id=torch.cuda.current_device() if device.startswith("cuda") else None)
        return model, "fsdp1"


def run(cfg: TrainConfig, args: argparse.Namespace, *, backend: str | None = None, rank: int | None = None,
        world_size: int | None = None, runs_root: str = "runs") -> dict[str, Any] | None:
    info = init_distributed(backend, rank, world_size)
    cfg.device = info.device
    torch.manual_seed(cfg.seed)
    pr_batch, global_batch = per_rank_batch(args.scaling, info.world_size, args.global_batch, args.per_rank_batch)
    cfg.batch_size = pr_batch
    cfg.seed = cfg.seed + 1000 * info.rank

    model_id = resolve_model_id(cfg.model)
    tok = None if cfg.model == TINY_RANDOM else build_tokenizer(model_id)
    pad_id = 0 if tok is None else tok.pad_token_id
    samples = load_samples(cfg, tok)
    # Build on CPU in fp32 (FSDP owns the dtype via its mixed-precision policy), then shard.
    build_cfg = TrainConfig(**{**cfg.to_dict(), "device": "cpu", "precision": "fp32" if cfg.precision != "fp32" else "fp32"})
    model, minfo = build_model(build_cfg)
    model, api = wrap_fsdp(model, cfg, info.device, reshard_after_forward=not args.no_reshard)
    if info.device.startswith("cuda"):
        model.to(info.device)

    t0 = time.perf_counter()
    summary = train_loop(model, cfg, samples, pad_id)
    summary["wall_s"] = time.perf_counter() - t0
    probes: dict[str, float] = {}
    if args.probe_comm and torch.cuda.is_available():
        opt = make_optimizer(model, 0.0)
        probes.update(comm_kernel_time(model, iter_batches(samples, cfg, pad_id),
                                       lambda b: optimizer_step(model, opt, [b], cfg, PhaseTimer())))
    local = {"rank": info.rank, "real_tokens_per_s": summary["real_tokens_per_s"], "mean_step_s": summary["mean_step_s"],
             "peak_allocated_gib": summary["peak_allocated_gib"], "peak_reserved_gib": summary["peak_reserved_gib"],
             "final_loss": summary["final_loss"], **probes}
    gathered = all_gather_object(local, info.world_size)
    result = None
    if info.rank == 0:
        agg_tps = sum(g["real_tokens_per_s"] for g in gathered)
        result = {
            "strategy": api, "scaling": args.scaling, "world_size": info.world_size, "backend": info.backend,
            "global_batch": global_batch, "per_rank_batch": pr_batch, "grad_accum": cfg.grad_accum,
            "reshard_after_forward": not args.no_reshard,
            "aggregate_real_tokens_per_s": agg_tps, "per_rank": gathered,
            "mean_step_s": max(g["mean_step_s"] for g in gathered),
            "peak_allocated_gib_max": max(g["peak_allocated_gib"] for g in gathered),
            "model_info": minfo, "config": cfg.to_dict(), "losses": summary["losses"], "phase_share": summary["phase_share"],
        }
        if runs_root is not None:
            with Run(LAB, f"{api}_w{info.world_size}_{args.scaling}", {**cfg.to_dict(), "scaling": args.scaling, "world_size": info.world_size},
                     root=runs_root, model={"id": model_id, "parallelism": f"{api.upper()} x{info.world_size}"},
                     workload={"dataset": cfg.dataset, "global_batch": global_batch, "per_rank_batch": pr_batch, "seq_len": cfg.seq_len},
                     execution={"steps": cfg.steps, "warmup_steps": cfg.warmup_steps}) as r:
                r.update(model=minfo)
                r.save_summary(result)
                result["run_dir"] = str(r.dir)
            print(f"[{api} w={info.world_size} {args.scaling}] aggregate {agg_tps:.0f} tok/s, slowest-rank step {result['mean_step_s']*1000:.1f} ms, "
                  f"peak alloc {result['peak_allocated_gib_max']:.2f} GiB, probes {probes}")
    dist.barrier()
    dist.destroy_process_group()
    return result


def main(argv=None) -> int:
    p = build_parser()
    p.add_argument("--no-reshard", action="store_true", help="keep gathered params after forward (ZeRO-2-like, more memory, less comm)")
    a = p.parse_args(argv)
    cfg = TrainConfig(model=a.model, dataset=a.dataset, precision=a.precision, grad_checkpointing=a.grad_checkpointing,
                      grad_accum=a.grad_accum, packing=a.packing, lora=a.lora, seq_len=a.seq_len, steps=a.steps,
                      warmup_steps=a.warmup_steps, lr=a.lr, n_samples=a.n_samples, seed=a.seed)
    if a.smoke:
        cfg.steps, cfg.warmup_steps, cfg.seq_len, cfg.n_samples = 6, 2, 128, 64
    run(cfg, a, runs_root=a.runs_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
