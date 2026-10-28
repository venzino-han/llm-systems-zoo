"""L07.7 — Ray Train `TorchTrainer`: orchestration vs computation.

Ray Train starts N workers (processes, optionally with GPUs), sets up the torch.distributed
process group, and collects metrics/checkpoints. The *math* — forward, backward, gradient
all-reduce — is plain PyTorch DDP produced by `ray.train.torch.prepare_model`. This script trains a
tiny causal transformer on synthetic tokens so it runs on CPU, reports per-worker throughput, and
checks that the workers end with identical parameters (the DDP all-reduce invariant).

Run:  python distributed/ray_train/torch_trainer.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ray  # noqa: E402

from distributed.ray_core._common import cuda_available, env_summary, init_ray  # noqa: E402
from llm_systems_zoo import Run  # noqa: E402


def build_model(vocab: int, d_model: int, n_layers: int, seq_len: int):
    import torch
    from torch import nn

    class TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(seq_len, d_model)
            layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=4 * d_model, batch_first=True, dropout=0.0)
            self.blocks = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Linear(d_model, vocab, bias=False)

        def forward(self, x):
            t = x.shape[1]
            h = self.emb(x) + self.pos(torch.arange(t, device=x.device))
            mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
            return self.head(self.blocks(h, mask=mask, is_causal=True))

    return TinyLM()


def param_digest(model) -> str:
    import torch

    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().float().cpu().numpy().tobytes())
    _ = torch
    return h.hexdigest()[:12]


def train_loop_per_worker(config: dict[str, Any]) -> None:
    import torch
    from ray import train
    from ray.train import Checkpoint
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(config["seed"])
    rank = train.get_context().get_world_rank()
    world = train.get_context().get_world_size()
    g = torch.Generator().manual_seed(config["seed"] + rank)
    data = torch.randint(0, config["vocab"], (config["n_samples"], config["seq_len"] + 1), generator=g)
    loader = DataLoader(TensorDataset(data[:, :-1], data[:, 1:]), batch_size=config["batch_size"], shuffle=True)
    loader = train.torch.prepare_data_loader(loader)  # adds DistributedSampler + device placement
    model = build_model(config["vocab"], config["d_model"], config["n_layers"], config["seq_len"])
    model = train.torch.prepare_model(model)  # wraps in DDP, moves to the worker's device
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()
    tokens = 0
    t0 = time.perf_counter()
    for epoch in range(config["epochs"]):
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        last = 0.0
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()  # DDP all-reduces gradients here
            opt.step()
            last = loss.item()
            tokens += x.numel()
        dt = time.perf_counter() - t0
        digest = param_digest(model.module if hasattr(model, "module") else model)
        digests: list = [None] * world
        if world > 1:
            torch.distributed.all_gather_object(digests, digest)  # DDP invariant: every rank holds the same weights
        else:
            digests = [digest]
        metrics = {"epoch": epoch, "loss": last, "tokens_per_s_per_worker": tokens / dt, "rank": rank, "world": world, "param_digest": digest, "params_identical_across_workers": len(set(digests)) == 1}
        ckpt = None
        if epoch == config["epochs"] - 1 and rank == 0:
            tmp = tempfile.mkdtemp()
            torch.save((model.module if hasattr(model, "module") else model).state_dict(), os.path.join(tmp, "model.pt"))
            ckpt = Checkpoint.from_directory(tmp)
        train.report(metrics, checkpoint=ckpt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=2048)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_samples = 2, 512
    use_gpu = args.use_gpu and cuda_available()
    cfg = {"vocab": 256, "seq_len": args.seq_len, "d_model": args.d_model, "n_layers": args.n_layers, "n_samples": args.n_samples, "batch_size": args.batch_size, "epochs": args.epochs, "lr": 1e-3, "seed": 0}

    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    with Run("L07_ray_systems", "ray_train", {**cfg, "workers": args.workers, "use_gpu": use_gpu}, workload=cfg) as run:
        init_ray(num_cpus=max(4, args.workers * 2), num_gpus=1 if use_gpu else 0)
        run.update(software={"ray_env": env_summary()})
        try:
            trainer = TorchTrainer(
                train_loop_per_worker,
                train_loop_config=cfg,
                scaling_config=ScalingConfig(num_workers=args.workers, use_gpu=use_gpu, resources_per_worker={"GPU": 1.0 / args.workers} if use_gpu else None),
                run_config=RunConfig(name="l07_ray_train", storage_path=str(run.dir.resolve() / "ray_results")),
            )
            t0 = time.perf_counter()
            result = trainer.fit()
            wall = time.perf_counter() - t0
        finally:
            ray.shutdown()
        m = dict(result.metrics)
        summary = {"final_metrics_rank0": m, "wall_s": wall, "checkpoint": str(result.checkpoint.path) if result.checkpoint else None, "workers": args.workers, "use_gpu": use_gpu}
        run.save_summary(summary)
        print(f"workers={args.workers} gpu={use_gpu} epochs={args.epochs} loss={m.get('loss'):.3f} tok/s/worker={m.get('tokens_per_s_per_worker'):.0f} digest={m.get('param_digest')} identical_across_workers={m.get('params_identical_across_workers')} wall={wall:.1f}s")
        print("run dir:", run.dir)


if __name__ == "__main__":
    main()
