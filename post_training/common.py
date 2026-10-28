"""Shared helpers for the L06 scripts: datasets, model/LoRA construction, bootstrap CI, memory."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from llm_systems_zoo.registry import MODELS
from post_training.rewards import SYSTEM_PROMPT, gold_answer

LAB = "L06_post_training"
ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def resolve_model_id(name: str) -> str:
    return MODELS[name].hf_id if name in MODELS else name


def lora_config(r: int):
    from peft import LoraConfig

    return LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules=ALL_LINEAR, task_type="CAUSAL_LM")


def bootstrap_ci(values: list[float] | np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return {"mean": float(arr.mean()), "ci_low": float(np.percentile(means, 100 * alpha / 2)),
            "ci_high": float(np.percentile(means, 100 * (1 - alpha / 2))), "n": int(arr.size)}


def gsm8k_rows(split: str, n: int | None, seed: int = 0) -> list[dict[str, Any]]:
    """GRPO/eval rows: conversational prompt + normalised gold answer."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if n:
        idx = idx[:n]
    rows = []
    for i in idx:
        ex = ds[i]
        rows.append({
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ex["question"]}],
            "answer": gold_answer(ex["answer"]),
            "question": ex["question"],
        })
    return rows


def sft_dataset(n: int | None, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train").shuffle(seed=seed)
    return ds.select(range(min(n, len(ds)))) if n else ds


def dpo_dataset(n: int | None, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train").shuffle(seed=seed)
    return ds.select(range(min(n, len(ds)))) if n else ds


def cuda_peak_gib() -> dict[str, float]:
    try:
        import torch

        if torch.cuda.is_available():
            return {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    except Exception:
        pass
    return {"peak_allocated_gib": 0.0, "peak_reserved_gib": 0.0}


def reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def common_args(p, default_steps: int = 20):
    p.add_argument("--model", default="small", help="registry key or HF id")
    p.add_argument("--lora", type=int, default=16, help="LoRA rank; 0 = full fine-tuning")
    p.add_argument("--steps", type=int, default=default_steps)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny model, 3 steps")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out", default="reports/L06_post_training/results")
    p.add_argument("--name", default=None)
    return p


def apply_smoke(a) -> None:
    if a.smoke:
        a.model, a.steps, a.batch_size, a.n_samples, a.max_length = "tiny", 3, 2, 16, 256
