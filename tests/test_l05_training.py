"""L05 training loop tests: resume equality, DDP on gloo, scaling arithmetic, table rendering.
CPU only; the only download-dependent test is marked slow."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from training.common import (
    TINY_RANDOM,
    TrainConfig,
    build_model,
    iter_batches,
    load_samples,
    train_loop,
)
from training.distributed.dist_utils import free_port, per_rank_batch
from training.train_efficiency import SWEEP_GRID, format_table, resume_test, run_once


def _cfg(**kw) -> TrainConfig:
    base = dict(model=TINY_RANDOM, dataset="synthetic", precision="fp32", batch_size=2, seq_len=64,
                steps=6, warmup_steps=2, n_samples=64, device="cpu")
    return TrainConfig(**{**base, **kw})


def test_train_loop_runs_and_reports_tokens(tmp_path: Path):
    cfg = _cfg()
    model, info = build_model(cfg)
    samples = load_samples(cfg)
    s = train_loop(model, cfg, samples, pad_id=0)
    assert len(s["losses"]) == 6 and all(torch.isfinite(torch.tensor(s["losses"])))
    assert s["measured_steps"] == 4 and s["real_tokens_per_s"] > 0
    assert 0.0 <= s["padding_fraction"] < 1.0
    assert set(s["phase_share"]) == {"forward", "backward", "optimizer"}
    assert info["n_trainable"] == info["n_params"] and info["est_static_gib"] > 0


def test_packing_reduces_padding_fraction_in_loop():
    cfg = _cfg()
    samples = load_samples(cfg)
    padded = next(iter_batches(samples, cfg, 0))
    packed = next(iter_batches(samples, TrainConfig(**{**cfg.to_dict(), "packing": True}), 0))
    assert packed.n_pad_tokens / packed.input_ids.numel() < padded.n_pad_tokens / padded.input_ids.numel()
    assert packed.input_ids.shape == (2, 64)


@pytest.mark.parametrize("packing", [False, True])
def test_checkpoint_resume_matches_continuous_run(tmp_path: Path, packing: bool):
    res = resume_test(_cfg(steps=8, packing=packing), tmp_path)
    assert res["passed"], res["max_abs_loss_diff"]
    assert res["checkpoint_step"] == 4 and len(res["resumed_losses"]) == 8


def test_grad_accum_changes_global_batch_not_loss_shape():
    cfg = _cfg(batch_size=1, grad_accum=2, steps=3, warmup_steps=1)
    assert cfg.global_batch == 2
    model, _ = build_model(cfg)
    s = train_loop(model, cfg, load_samples(cfg), 0)
    assert len(s["losses"]) == 3


def test_run_once_writes_summary(tmp_path: Path):
    s = run_once(_cfg(steps=3, warmup_steps=1), "unit", root=str(tmp_path), collect_env=False)
    assert Path(s["run_dir"]).joinpath("summary.json").exists()
    assert Path(s["run_dir"]).joinpath("metrics.jsonl").exists()


def test_sweep_table_renders_failures_and_successes():
    rows = [{"name": "ok", "config": _cfg().to_dict(), "phase_share": {"forward": 0.3, "backward": 0.5, "optimizer": 0.2},
             "model_info": {"trainable_frac": 1.0, "est_static_gib": 1.0}, "real_tokens_per_s": 10.0, "padding_fraction": 0.1,
             "peak_allocated_gib": 1.0, "peak_reserved_gib": 2.0, "mean_step_s": 0.1, "mean_loss_last5": 1.0},
            {"name": "oom", "config": _cfg().to_dict(), "error": "OutOfMemoryError: CUDA out of memory"}]
    md = format_table(rows)
    assert "| ok |" in md and "FAILED" in md
    assert {g["name"] for g in SWEEP_GRID} >= {"fp32", "bf16_autocast", "qlora16"}


def test_per_rank_batch_arithmetic():
    assert per_rank_batch("strong", 4, 16, None) == (4, 16)
    assert per_rank_batch("weak", 4, None, 3) == (3, 12)
    with pytest.raises(AssertionError):
        per_rank_batch("strong", 3, 16, None)


# ---- DDP on gloo ---------------------------------------------------------------------------
def _ddp_worker(rank: int, world: int, port: int, out_dir: str):
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "CUDA_VISIBLE_DEVICES": ""})
    import argparse

    from training.distributed.ddp_train import run

    args = argparse.Namespace(scaling="strong", global_batch=4, per_rank_batch=None, bucket_cap_mb=25, probe_comm=True)
    res = run(_cfg(steps=4, warmup_steps=1), args, backend="gloo", rank=rank, world_size=world, runs_root=None)
    if rank == 0:
        torch.save(res, os.path.join(out_dir, "ddp_result.pt"))


def test_ddp_gloo_two_processes(tmp_path: Path):
    torch.multiprocessing.spawn(_ddp_worker, args=(2, free_port(), str(tmp_path)), nprocs=2, join=True)
    res = torch.load(tmp_path / "ddp_result.pt", weights_only=False)
    assert res["world_size"] == 2 and res["per_rank_batch"] == 2 and res["global_batch"] == 4
    assert len(res["per_rank"]) == 2 and all(torch.isfinite(torch.tensor(r["final_loss"])) for r in res["per_rank"])
    assert "exposed_comm_s" in res["per_rank"][0]


@pytest.mark.slow
def test_tiny_pretrained_model_two_steps_cpu(tmp_path: Path):
    cfg = TrainConfig(model="tiny", dataset="synthetic", precision="fp32", batch_size=1, seq_len=32, steps=2,
                      warmup_steps=1, n_samples=8, device="cpu")
    s = run_once(cfg, "slow_tiny", root=str(tmp_path), collect_env=False)
    assert s["final_loss"] > 0
