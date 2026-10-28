"""Attention kernel comparison: PyTorch SDPA backends, FlashInfer, FlashAttention-3, and a naive
materialised-scores implementation, across sequence lengths and head dims.

All candidates compute the same causal attention `softmax(QK^T / sqrt(d)) V` and are checked
against the SDPA *math* backend (fp32 reference). What differs is how the score matrix is
handled:

* naive / math: materialise the (L x L) score matrix -> O(L^2) memory, multiple passes.
* flash / efficient / cuDNN / FlashInfer / FA3: tiled online-softmax -> O(L) extra memory, one pass.

Run from the repo root:

    .venv/bin/python inference/kernels/attention_bench.py --smoke
    .venv/bin/python inference/kernels/attention_bench.py --seq 512 1024 2048 4096 8192 --head-dim 64 128
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from llm_systems_zoo import Run
from llm_systems_zoo.plotting import plot_lines


def naive_attention(q, k, v, causal=True):
    """Explicit scores in fp32: the textbook formula and the memory you pay for it."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        L, S = scores.shape[-2:]
        mask = torch.ones(L, S, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def sdpa_with_backend(backend_name: str):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    be = getattr(SDPBackend, backend_name)

    def fn(q, k, v, causal=True):
        with sdpa_kernel(be):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    return fn


def flashinfer_attention():
    import flashinfer

    def fn(q, k, v, causal=True):
        # flashinfer wants [L, H, D]; our tensors are [B, H, L, D] with B == 1
        o = flashinfer.single_prefill_with_kv_cache(
            q[0].transpose(0, 1).contiguous(), k[0].transpose(0, 1).contiguous(), v[0].transpose(0, 1).contiguous(), causal=causal
        )
        return o.transpose(0, 1).unsqueeze(0)

    return fn


def fa3_attention():
    # The installed flash-attn-3 wheel is the Hopper (sm_90) build. On any other GPU the launch
    # fails inside C++ with "no kernel image is available" and *aborts the process*, so gate it
    # here instead of relying on try/except.
    cc = torch.cuda.get_device_capability()
    if cc[0] != 9:
        raise RuntimeError(f"flash-attn-3 Hopper build cannot run on sm_{cc[0]}{cc[1]} (no kernel image)")
    import flash_attn_interface as fai

    def fn(q, k, v, causal=True):
        o = fai.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal)
        if isinstance(o, tuple):
            o = o[0]
        return o.transpose(1, 2)

    return fn


CANDIDATES = {
    "naive_fp32": lambda: naive_attention,
    "sdpa_math": lambda: sdpa_with_backend("MATH"),
    "sdpa_efficient": lambda: sdpa_with_backend("EFFICIENT_ATTENTION"),
    "sdpa_flash": lambda: sdpa_with_backend("FLASH_ATTENTION"),
    "sdpa_cudnn": lambda: sdpa_with_backend("CUDNN_ATTENTION"),
    "flashinfer_prefill": flashinfer_attention,
    "flash_attn_3": fa3_attention,
}


def _time_and_mem(fn, q, k, v, warmup=5, rep=20):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        out = fn(q, k, v)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / rep
    peak_extra = torch.cuda.max_memory_allocated() - base
    return ms, peak_extra, out


def run_grid(seqs, head_dims, heads, batch, dtype, candidates, device="cuda"):
    rows = []
    status = {}
    for d in head_dims:
        for L in seqs:
            q = torch.randn(batch, heads, L, d, device=device, dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            flops = 4 * batch * heads * L * L * d / 2  # causal: half the score matrix, QK^T and PV
            ref = None
            try:
                _, _, ref = _time_and_mem(sdpa_with_backend("MATH"), q, k, v, warmup=1, rep=1)
            except Exception:
                pass
            for name in candidates:
                entry = {"seq": L, "head_dim": d, "heads": heads, "batch": batch, "candidate": name}
                try:
                    fn = CANDIDATES[name]()
                    ms, peak, out = _time_and_mem(fn, q, k, v)
                    entry.update({"ms": ms, "peak_extra_mib": peak / 2**20, "tflops": flops / (ms * 1e-3) / 1e12})
                    if ref is not None:
                        entry["max_abs_err_vs_math"] = (out.float() - ref.float()).abs().max().item()
                    status.setdefault(name, "ok")
                except Exception as e:  # OOM or unsupported backend: record it, keep going
                    msg = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
                    entry["error"] = msg
                    status[name] = msg
                    torch.cuda.empty_cache()
                rows.append(entry)
            del q, k, v, ref
            torch.cuda.empty_cache()
    return rows, status


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    p.add_argument("--head-dim", type=int, nargs="+", default=[64, 128])
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--report-dir", type=Path, default=Path("reports/L08_inference_optimization/results"))
    args = p.parse_args()
    if args.smoke:
        args.seq, args.head_dim, args.heads = [256, 1024], [64], 8
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    dtype = getattr(torch, args.dtype)
    cfg = {k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(args).items()}
    with Run("L08_inference_optimization", "attention_bench", cfg, execution={"warmup": 5, "rep": 20, "timer": "cuda events"}) as run:
        rows, status = run_grid(args.seq, args.head_dim, args.heads, args.batch, dtype, args.candidates)
        run.save_json("bench.json", rows)
        run.save_json("backend_status.json", status)
        for d in args.head_dim:
            series = {}
            for name in args.candidates:
                ys = [next((r["ms"] for r in rows if r["candidate"] == name and r["seq"] == L and r["head_dim"] == d and "ms" in r), float("nan")) for L in args.seq]
                if any(y == y for y in ys):
                    series[name] = ys
            plot_lines(args.seq, series, xlabel="sequence length", ylabel="latency (ms)", title=f"causal attention, {args.heads} heads, d={d}, {args.dtype}", path=run.path(f"attention_latency_d{d}.png"), logx=True, logy=True)
            mem = {}
            for name in args.candidates:
                ys = [next((r["peak_extra_mib"] for r in rows if r["candidate"] == name and r["seq"] == L and r["head_dim"] == d and "ms" in r), float("nan")) for L in args.seq]
                if any(y == y for y in ys):
                    mem[name] = ys
            plot_lines(args.seq, mem, xlabel="sequence length", ylabel="peak extra memory (MiB)", title=f"attention workspace, d={d}", path=run.path(f"attention_memory_d{d}.png"), logx=True, logy=True)
        summary = {"device": torch.cuda.get_device_name(0), "backend_status": status, "rows": rows}
        run.save_summary(summary)
        if not args.smoke:
            import shutil

            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / "attention_bench.json").write_text(json.dumps(summary, indent=2))
            for f in run.dir.glob("attention_*.png"):
                shutil.copy(f, args.report_dir / f.name)
        print("run_dir", run.dir)
        print(json.dumps(status, indent=2))
        for r in rows:
            if "ms" in r:
                print(f"d={r['head_dim']:4d} L={r['seq']:5d} {r['candidate']:20s} {r['ms']:8.3f} ms  {r['peak_extra_mib']:9.1f} MiB  err={r.get('max_abs_err_vs_math', float('nan')):.2e}")


if __name__ == "__main__":
    main()
