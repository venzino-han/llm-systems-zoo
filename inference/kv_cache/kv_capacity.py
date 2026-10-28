"""L02 — how many KV tokens fit on a GPU, and why GQA matters.

    kv_bytes_per_token = 2 × L × a_kv × d × p
    kv_budget          = gpu_memory × utilisation − weights − runtime_overhead
    max_tokens         = kv_budget / kv_bytes_per_token
    max_concurrency    = max_tokens / (prompt_len + gen_len)

The runtime overhead term (CUDA context, activation workspace, CUDA-graph pools) is a
configurable constant because it is measured, not derived. vLLM additionally reserves memory
for activations during profiling and rounds the budget down to whole blocks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from fundamentals.memory_calculator import DTYPE_BYTES, GIB, ModelConfig, get_config


def kv_bytes_per_token(cfg: ModelConfig, kv_dtype: str = "bf16") -> float:
    return cfg.kv_bytes_per_token(kv_dtype)


def kv_budget_bytes(cfg: ModelConfig, gpu_memory_gib: float, utilisation: float = 0.9,
                    weight_dtype: str = "bf16", overhead_gib: float = 1.0) -> float:
    return gpu_memory_gib * GIB * utilisation - cfg.n_params * DTYPE_BYTES[weight_dtype] - overhead_gib * GIB


def max_kv_tokens(cfg: ModelConfig, gpu_memory_gib: float, utilisation: float = 0.9, weight_dtype: str = "bf16",
                  kv_dtype: str = "bf16", overhead_gib: float = 1.0) -> int:
    budget = kv_budget_bytes(cfg, gpu_memory_gib, utilisation, weight_dtype, overhead_gib)
    return max(int(budget // kv_bytes_per_token(cfg, kv_dtype)), 0)


def max_concurrency(cfg: ModelConfig, gpu_memory_gib: float, prompt_len: int, gen_len: int, **kw) -> int:
    return max_kv_tokens(cfg, gpu_memory_gib, **kw) // (prompt_len + gen_len)


def mha_equivalent(cfg: ModelConfig) -> ModelConfig:
    """Same model with one KV head per query head (what the cache would cost without GQA)."""
    return replace(cfg, n_kv_heads=cfg.n_heads, name=cfg.name + " (MHA-equivalent)")


def capacity_table(models: list[str], gpu_memory_gib: float, utilisation: float, prompt_len: int, gen_len: int,
                   weight_dtype: str = "bf16", kv_dtype: str = "bf16", overhead_gib: float = 1.0) -> list[dict]:
    rows = []
    for m in models:
        cfg = get_config(m)
        for variant in (cfg, mha_equivalent(cfg)):
            tokens = max_kv_tokens(variant, gpu_memory_gib, utilisation, weight_dtype, kv_dtype, overhead_gib)
            rows.append({
                "model": variant.name,
                "kv_heads": variant.n_kv_heads,
                "q_heads": variant.n_heads,
                "kv_kib_per_token": kv_bytes_per_token(variant, kv_dtype) / 1024,
                "weights_gib": variant.n_params * DTYPE_BYTES[weight_dtype] / GIB,
                "kv_budget_gib": kv_budget_bytes(variant, gpu_memory_gib, utilisation, weight_dtype, overhead_gib) / GIB,
                "max_kv_tokens": tokens,
                "max_concurrency": tokens // (prompt_len + gen_len),
            })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="KV-cache capacity model.")
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B"])
    ap.add_argument("--gpu-gib", type=float, default=48.0)
    ap.add_argument("--utilisation", type=float, default=0.9)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--weight-dtype", default="bf16")
    ap.add_argument("--kv-dtype", default="bf16")
    ap.add_argument("--overhead-gib", type=float, default=1.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rows = capacity_table(a.models, a.gpu_gib, a.utilisation, a.prompt_len, a.gen_len, a.weight_dtype, a.kv_dtype, a.overhead_gib)
    if a.json:
        print(json.dumps(rows, indent=2))
        return
    print(f"GPU {a.gpu_gib} GiB × {a.utilisation} utilisation, weights {a.weight_dtype}, KV {a.kv_dtype}, "
          f"overhead {a.overhead_gib} GiB, request = {a.prompt_len}+{a.gen_len} tokens")
    print(f"{'model':<48}{'kv/q heads':>11}{'KiB/tok':>9}{'weights':>9}{'KV GiB':>8}{'max tokens':>12}{'max conc':>10}")
    for r in rows:
        print(f"{r['model']:<48}{str(r['kv_heads']) + '/' + str(r['q_heads']):>11}{r['kv_kib_per_token']:>9.1f}"
              f"{r['weights_gib']:>9.2f}{r['kv_budget_gib']:>8.2f}{r['max_kv_tokens']:>12,}{r['max_concurrency']:>10,}")


if __name__ == "__main__":
    main()
