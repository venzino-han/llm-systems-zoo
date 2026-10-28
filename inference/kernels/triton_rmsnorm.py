"""RMSNorm (and fused residual-add + RMSNorm) in Triton, with correctness check and benchmark.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * w   (Zhang & Sennrich 2019; used by Llama/Qwen/Mistral)

Why it is worth a kernel: in eager PyTorch the Llama reference implementation launches ~6 small
kernels per call (pow, mean, add, rsqrt, mul, mul) and each decode step calls it 2x per layer.
At batch 1 the op is pure memory traffic, so fusing it into one read + one write is the whole win.
The *fused residual* variant goes one step further and also folds the `h = h + residual` that
precedes every norm in a transformer block, saving another full read and write of the hidden
state. Production engines (vLLM, TensorRT-LLM, SGLang) ship exactly this fused op.

Run from the repo root:

    .venv/bin/python inference/kernels/triton_rmsnorm.py --smoke
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from llm_systems_zoo import Run
from llm_systems_zoo.plotting import plot_lines

if HAS_TRITON:

    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(out_ptr + row * stride + offs, y.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _fused_add_rmsnorm_kernel(
        x_ptr, res_ptr, w_ptr, out_ptr, stride, n_cols, eps, BLOCK_SIZE: tl.constexpr
    ):
        # out = rmsnorm(x + residual); residual <- x + residual   (vLLM's fused_add_rms_norm)
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
        h = x + r
        tl.store(res_ptr + row * stride + offs, h.to(res_ptr.dtype.element_ty), mask=mask)
        var = tl.sum(h * h, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * stride + offs, (h * rstd * w).to(out_ptr.dtype.element_ty), mask=mask)


def _launch_cfg(n_cols: int):
    block = triton.next_power_of_2(n_cols)
    num_warps = 4 if block < 2048 else (8 if block < 8192 else 16)
    return block, num_warps


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    out = torch.empty_like(x2)
    block, nw = _launch_cfg(x2.shape[1])
    _rmsnorm_kernel[(x2.shape[0],)](x2, weight, out, x2.stride(0), x2.shape[1], eps, BLOCK_SIZE=block, num_warps=nw)
    return out.reshape(shape)


def triton_fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (normed, new_residual). `residual` is updated in place, as in vLLM."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    r2 = residual.reshape(-1, shape[-1])
    assert r2.is_contiguous(), "residual must be contiguous (it is updated in place)"
    out = torch.empty_like(x2)
    block, nw = _launch_cfg(x2.shape[1])
    _fused_add_rmsnorm_kernel[(x2.shape[0],)](
        x2, r2, weight, out, x2.stride(0), x2.shape[1], eps, BLOCK_SIZE=block, num_warps=nw
    )
    return out.reshape(shape), residual


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The HF Llama/Qwen RMSNorm, verbatim: fp32 statistics, cast back, then scale."""
    dtype = x.dtype
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return weight * h.to(dtype)


def reference_fused(x, residual, weight, eps=1e-6):
    h = x + residual
    return reference_rmsnorm(h, weight, eps), h


def check_correctness(
    shapes=((4, 17), (8, 896), (3, 1536), (2, 4096), (1, 8192)),
    dtypes=(torch.float32, torch.float16, torch.bfloat16),
    device="cuda",
) -> list[dict]:
    results = []
    for dtype in dtypes:
        for shape in shapes:
            x = torch.randn(*shape, device=device, dtype=dtype)
            r = torch.randn(*shape, device=device, dtype=dtype)
            w = (1 + 0.1 * torch.randn(shape[-1], device=device)).to(dtype)
            tol = 1e-5 if dtype == torch.float32 else 2e-2
            got = triton_rmsnorm(x, w)
            ref = reference_rmsnorm(x, w)
            err = (got.float() - ref.float()).abs().max().item()
            ok = torch.allclose(got.float(), ref.float(), atol=tol, rtol=tol)
            got_f, res_f = triton_fused_add_rmsnorm(x.clone(), r.clone(), w)
            ref_f, ref_res = reference_fused(x, r, w)
            err_f = max((got_f.float() - ref_f.float()).abs().max().item(), (res_f.float() - ref_res.float()).abs().max().item())
            ok_f = torch.allclose(got_f.float(), ref_f.float(), atol=tol, rtol=tol) and torch.allclose(res_f.float(), ref_res.float(), atol=tol, rtol=tol)
            results.append({"dtype": str(dtype).split(".")[-1], "shape": list(shape), "max_abs_err": err, "ok": bool(ok), "fused_max_abs_err": err_f, "fused_ok": bool(ok_f)})
    return results


def benchmark(hidden_sizes: list[int], rows: int, dtype: torch.dtype, device="cuda", compiled=True) -> list[dict]:
    from triton.testing import do_bench

    out = []
    compiled_ref = torch.compile(reference_rmsnorm, dynamic=True) if compiled else None
    compiled_fused = torch.compile(reference_fused, dynamic=True) if compiled else None
    for h in hidden_sizes:
        x = torch.randn(rows, h, device=device, dtype=dtype)
        r = torch.randn(rows, h, device=device, dtype=dtype)
        w = torch.ones(h, device=device, dtype=dtype)
        entry = {"hidden": h, "rows": rows, "dtype": str(dtype).split(".")[-1]}
        cands = {
            "triton_rmsnorm": partial(triton_rmsnorm, x, w),
            "torch_eager_rmsnorm": partial(reference_rmsnorm, x, w),
            "triton_fused_add_rmsnorm": partial(triton_fused_add_rmsnorm, x, r, w),
            "torch_eager_add_rmsnorm": partial(reference_fused, x, r, w),
        }
        if compiled:
            compiled_ref(x, w)
            compiled_fused(x, r, w)
            cands["torch_compiled_rmsnorm"] = partial(compiled_ref, x, w)
            cands["torch_compiled_add_rmsnorm"] = partial(compiled_fused, x, r, w)
        for name, fn in cands.items():
            entry[f"{name}_ms"] = do_bench(fn, warmup=10, rep=50)
        out.append(entry)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", type=int, nargs="+", default=[896, 1536, 2048, 3584, 4096, 8192])
    p.add_argument("--rows", type=int, default=4096, help="tokens in the batch (batch*seq)")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--report-dir", type=Path, default=Path("reports/L08_inference_optimization/results"))
    args = p.parse_args()
    if args.smoke:
        args.hidden, args.rows = [896, 4096], 1024
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    dtype = getattr(torch, args.dtype)
    cfg = {"hidden": args.hidden, "rows": args.rows, "dtype": args.dtype, "compile": not args.no_compile, "smoke": args.smoke}
    with Run("L08_inference_optimization", "triton_rmsnorm", cfg, execution={"warmup": 10, "rep": 50}) as run:
        corr = check_correctness()
        run.save_json("correctness.json", corr)
        bad = [c for c in corr if not (c["ok"] and c["fused_ok"])]
        if bad:
            raise RuntimeError(f"correctness failed: {bad}")
        bench = benchmark(args.hidden, args.rows, dtype, compiled=not args.no_compile)
        run.save_json("bench.json", bench)
        keys = [k[:-3] for k in bench[0] if k.endswith("_ms")]
        plot_lines(args.hidden, {k: [b[f"{k}_ms"] for b in bench] for k in keys}, xlabel="hidden size",
                   ylabel="latency (ms)", title=f"RMSNorm, {args.rows} tokens, {args.dtype}", path=run.path("rmsnorm_latency.png"), logy=True)
        summary = {"device": torch.cuda.get_device_name(0), "correctness_ok": not bad, "bench": bench}
        run.save_summary(summary)
        if not args.smoke:
            import shutil

            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / "triton_rmsnorm.json").write_text(json.dumps(summary, indent=2))
            shutil.copy(run.path("rmsnorm_latency.png"), args.report_dir / "triton_rmsnorm_latency.png")
        print(json.dumps({"run_dir": str(run.dir), "correctness_ok": not bad}))
        for b in bench:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()})


if __name__ == "__main__":
    main()
