"""L01 — FLOPs, bytes moved, arithmetic intensity and a roofline helper.

Why this exists: "why is decode slow when the GPU is idle?" has a one-line answer once you
compute arithmetic intensity. Per generated token, decode touches every weight once
(N * bytes) and the whole KV cache, but performs only ~2N FLOPs — far below the ridge point
of any modern GPU, so decode is memory-bandwidth bound. Prefill processes s tokens per weight
read, so it is compute bound once s is a few hundred.

Formulas (N = non-embedding + embedding parameters, L = layers, h = d_model, s = context,
a_kv = KV heads, d = head_dim):

    forward FLOPs / token         2N + 4 L h s      (linear layers + QK^T and AV at context s)
    training FLOPs / token        3 × forward        (backward ≈ 2× forward)       ≈ 6N per token
    prefill FLOPs (s tokens)      2N s + 2 L h s^2   (causal attention: half of 4 L h s^2)
    decode FLOPs at position t    2N + 4 L h t
    decode bytes at position t    N p_w + b t kv_bytes_per_token  (weights + KV read once)
    arithmetic intensity          FLOPs / bytes
    roofline time                 max(FLOPs / peak_flops, bytes / bandwidth)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from fundamentals.memory_calculator import DTYPE_BYTES, ModelConfig, get_config


@dataclass(frozen=True)
class GPUSpec:
    """Nominal, datasheet-level numbers. Dense (non-sparse) tensor-core throughput.
    They are upper bounds used for roofline ratios, not measured values on this machine."""

    name: str
    peak_tflops_bf16: float
    bandwidth_gb_s: float
    memory_gib: float
    source: str


# Approximate public datasheet values (dense BF16/FP16 tensor-core, HBM/GDDR bandwidth).
KNOWN_GPUS: dict[str, GPUSpec] = {
    "RTX A6000": GPUSpec("RTX A6000", 155, 768, 48, "NVIDIA datasheet: 309.7 TFLOPS with sparsity → ~155 dense; 768 GB/s GDDR6"),
    "RTX PRO 6000 Blackwell": GPUSpec("RTX PRO 6000 Blackwell", 500, 1792, 96, "NVIDIA datasheet: ~1 PFLOPS FP16 w/ sparsity → ~500 dense (approx.); 1.79 TB/s GDDR7"),
    "A100 80GB SXM": GPUSpec("A100 80GB SXM", 312, 2039, 80, "NVIDIA datasheet: 312 TFLOPS BF16 dense; 2039 GB/s HBM2e"),
    "H100 SXM": GPUSpec("H100 SXM", 989, 3350, 80, "NVIDIA datasheet: 989 TFLOPS BF16 dense; 3.35 TB/s HBM3"),
}


def forward_flops_per_token(cfg: ModelConfig, context_len: int) -> float:
    return 2 * cfg.n_params + 4 * cfg.n_layers * cfg.d_model * context_len


def training_flops(cfg: ModelConfig, tokens: int, seq_len: int) -> float:
    """≈ 6 N D plus the attention term at the given sequence length."""
    return 3 * tokens * forward_flops_per_token(cfg, seq_len)


def prefill_flops(cfg: ModelConfig, prompt_len: int, batch: int = 1) -> float:
    linear = 2 * cfg.n_params * prompt_len
    attention = 2 * cfg.n_layers * cfg.d_model * prompt_len**2  # causal → half of the full s×s
    return batch * (linear + attention)


def decode_step_flops(cfg: ModelConfig, position: int, batch: int = 1) -> float:
    return batch * forward_flops_per_token(cfg, position)


def decode_step_bytes(cfg: ModelConfig, position: int, batch: int = 1, weight_dtype: str = "bf16", kv_dtype: str = "bf16") -> float:
    weights = cfg.n_params * DTYPE_BYTES[weight_dtype]
    kv = batch * position * cfg.kv_bytes_per_token(kv_dtype)
    return weights + kv


def prefill_bytes(cfg: ModelConfig, prompt_len: int, batch: int = 1, weight_dtype: str = "bf16", kv_dtype: str = "bf16") -> float:
    """Weights are read once per layer pass (not per token); the KV cache is written once."""
    return cfg.n_params * DTYPE_BYTES[weight_dtype] + batch * prompt_len * cfg.kv_bytes_per_token(kv_dtype)


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    return flops / bytes_moved


def ridge_point(gpu: GPUSpec) -> float:
    """FLOPs/byte above which a kernel is compute bound on this GPU."""
    return gpu.peak_tflops_bf16 * 1e12 / (gpu.bandwidth_gb_s * 1e9)


def roofline_time_s(flops: float, bytes_moved: float, gpu: GPUSpec) -> dict[str, float]:
    t_compute = flops / (gpu.peak_tflops_bf16 * 1e12)
    t_memory = bytes_moved / (gpu.bandwidth_gb_s * 1e9)
    bound = "compute" if t_compute >= t_memory else "memory"
    return {"t_compute_s": t_compute, "t_memory_s": t_memory, "t_roofline_s": max(t_compute, t_memory), "bound": bound,
            "intensity": arithmetic_intensity(flops, bytes_moved), "ridge": ridge_point(gpu)}


def decode_roofline(cfg: ModelConfig, batch: int, position: int, gpu: GPUSpec, weight_dtype: str = "bf16") -> dict[str, float]:
    f = decode_step_flops(cfg, position, batch)
    b = decode_step_bytes(cfg, position, batch, weight_dtype, weight_dtype)
    out = roofline_time_s(f, b, gpu)
    out["tokens_per_s_bound"] = batch / out["t_roofline_s"]
    return out


def prefill_roofline(cfg: ModelConfig, batch: int, prompt_len: int, gpu: GPUSpec, weight_dtype: str = "bf16") -> dict[str, float]:
    f = prefill_flops(cfg, prompt_len, batch)
    b = prefill_bytes(cfg, prompt_len, batch, weight_dtype, weight_dtype)
    out = roofline_time_s(f, b, gpu)
    out["tokens_per_s_bound"] = batch * prompt_len / out["t_roofline_s"]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="FLOPs / roofline estimates for an LLM on a GPU.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--gpu", default="RTX A6000", choices=sorted(KNOWN_GPUS))
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 64])
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cfg, gpu = get_config(args.model), KNOWN_GPUS[args.gpu]
    rows = []
    for b in args.batch:
        pre = prefill_roofline(cfg, b, args.prompt_len, gpu, args.dtype)
        dec = decode_roofline(cfg, b, args.prompt_len, gpu, args.dtype)
        rows.append({"batch": b, "prefill": pre, "decode": dec})
    if args.json:
        print(json.dumps({"model": cfg.name, "gpu": gpu.name, "ridge_flops_per_byte": ridge_point(gpu), "rows": rows}, indent=2))
        return
    print(f"{cfg.name} on {gpu.name} (nominal {gpu.peak_tflops_bf16} TFLOPS, {gpu.bandwidth_gb_s} GB/s, ridge {ridge_point(gpu):.0f} FLOP/B)")
    print(f"training FLOPs/token ≈ {3 * forward_flops_per_token(cfg, args.prompt_len) / 1e9:.2f} GFLOP (6N ≈ {6 * cfg.n_params / 1e9:.2f})")
    print(f"{'batch':>6} {'phase':>8} {'intensity':>10} {'bound':>8} {'t_roofline_ms':>14} {'tok/s bound':>12}")
    for r in rows:
        for phase in ("prefill", "decode"):
            x = r[phase]
            print(f"{r['batch']:>6} {phase:>8} {x['intensity']:>10.1f} {x['bound']:>8} {x['t_roofline_s'] * 1e3:>14.2f} {x['tokens_per_s_bound']:>12,.0f}")


if __name__ == "__main__":
    main()
