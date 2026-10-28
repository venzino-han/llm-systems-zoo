"""Educational row-wise softmax in Triton, with a correctness check and a benchmark.

What the kernel does: one Triton *program* (a block of threads) owns one row. It loads the
row once, computes `exp(x - max) / sum(exp(x - max))` in fp32 registers, and stores it. That is
the memory-bound ideal: 1 read + 1 write of the row, instead of the up-to-5 passes a naive
PyTorch composition (max, sub, exp, sum, div) would do when the ops are not fused.

This is the Triton tutorial design (one block per row, `BLOCK_SIZE = next_power_of_2(n_cols)`),
not what a production engine ships: engines fuse softmax *into* attention (FlashAttention's
online softmax) so the full score matrix never exists. See README.md section 1.

Run from the repo root:

    .venv/bin/python inference/kernels/triton_softmax.py --smoke
    .venv/bin/python inference/kernels/triton_softmax.py --cols 256 1024 4096 16384 --rows 4096
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
except Exception:  # pragma: no cover - CPU CI
    HAS_TRITON = False

from llm_systems_zoo import Run
from llm_systems_zoo.plotting import plot_lines

if HAS_TRITON:

    @triton.jit
    def _softmax_kernel(
        out_ptr,
        in_ptr,
        in_row_stride,
        out_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        row_start = in_ptr + row_idx * in_row_stride
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        # Load once; out-of-range lanes get -inf so they vanish in max and exp.
        row = tl.load(row_start + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.max(row, axis=0)
        num = tl.exp(row - row_max)
        den = tl.sum(num, axis=0)
        out = num / den
        tl.store(out_ptr + row_idx * out_row_stride + offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dimension. Returns the input dtype, computes in fp32."""
    if not HAS_TRITON:
        raise RuntimeError("triton is not available")
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    n_rows, n_cols = x2.shape
    block = triton.next_power_of_2(n_cols)
    # More warps for wider rows: 4 up to 2k columns, 8 up to 4k, 16 beyond.
    num_warps = 4 if block < 2048 else (8 if block < 4096 else 16)
    out = torch.empty_like(x2)
    _softmax_kernel[(n_rows,)](
        out, x2, x2.stride(0), out.stride(0), n_cols, BLOCK_SIZE=block, num_warps=num_warps
    )
    return out.reshape(shape)


def reference_softmax(x: torch.Tensor) -> torch.Tensor:
    """fp32 reference, cast back to the input dtype (what the kernel is compared against)."""
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """The unfused composition: 5 kernel launches, ~5 passes over memory in eager mode."""
    x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    z = x - x_max
    num = torch.exp(z)
    den = num.sum(dim=-1, keepdim=True)
    return num / den


def check_correctness(
    shapes=((4, 17), (8, 256), (3, 1000), (2, 4096), (1, 16384)),
    dtypes=(torch.float32, torch.float16, torch.bfloat16),
    device: str = "cuda",
) -> list[dict]:
    """Compare the Triton kernel with the fp32 reference across odd and power-of-two widths."""
    results = []
    for dtype in dtypes:
        for shape in shapes:
            x = torch.randn(*shape, device=device, dtype=dtype) * 3
            got = triton_softmax(x)
            ref = reference_softmax(x)
            tol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 2e-2}[dtype]
            max_err = (got.float() - ref.float()).abs().max().item()
            ok = torch.allclose(got.float(), ref.float(), atol=tol, rtol=tol)
            # rows sum to one up to the output dtype's rounding (bf16 keeps ~3 significant digits)
            ok = ok and torch.allclose(got.float().sum(-1), torch.ones(shape[:-1], device=device), atol=tol)
            results.append({"dtype": str(dtype).split(".")[-1], "shape": list(shape), "max_abs_err": max_err, "ok": bool(ok)})
    return results


def benchmark(
    cols: list[int], rows: int, dtype: torch.dtype, device: str = "cuda", compiled: bool = True
) -> list[dict]:
    """Latency (ms) and effective bandwidth for triton vs torch.softmax vs naive vs torch.compile."""
    from triton.testing import do_bench

    rows_out = []
    compiled_fn = torch.compile(naive_softmax, dynamic=True) if compiled else None
    for n in cols:
        x = torch.randn(rows, n, device=device, dtype=dtype)
        bytes_moved = 2 * x.numel() * x.element_size()  # one read + one write of the row
        entry = {"cols": n, "rows": rows, "dtype": str(dtype).split(".")[-1], "bytes_moved": bytes_moved}
        candidates = {
            "triton": partial(triton_softmax, x),
            "torch_softmax": partial(torch.softmax, x, dim=-1),
            "naive_eager": partial(naive_softmax, x),
        }
        if compiled_fn is not None:
            compiled_fn(x)  # warm-up / compile outside the timed region
            candidates["naive_compiled"] = partial(compiled_fn, x)
        for name, fn in candidates.items():
            ms = do_bench(fn, warmup=10, rep=50)
            entry[f"{name}_ms"] = ms
            entry[f"{name}_gbps"] = bytes_moved / (ms * 1e-3) / 1e9
        rows_out.append(entry)
    return rows_out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cols", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--no-compile", action="store_true", help="skip the torch.compile candidate")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--report-dir", type=Path, default=Path("reports/L08_inference_optimization/results"))
    args = p.parse_args()
    if args.smoke:
        args.cols, args.rows = [256, 1024, 4096], 1024
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    dtype = getattr(torch, args.dtype)
    cfg = {"cols": args.cols, "rows": args.rows, "dtype": args.dtype, "compile": not args.no_compile, "smoke": args.smoke}
    with Run("L08_inference_optimization", "triton_softmax", cfg, execution={"warmup": 10, "rep": 50, "timer": "triton.testing.do_bench"}) as run:
        correctness = check_correctness()
        bad = [r for r in correctness if not r["ok"]]
        run.save_json("correctness.json", correctness)
        if bad:
            raise RuntimeError(f"correctness failed: {bad}")
        bench = benchmark(args.cols, args.rows, dtype, compiled=not args.no_compile)
        run.save_json("bench.json", bench)
        series = {k: [b[f"{k}_gbps"] for b in bench] for k in ("triton", "torch_softmax", "naive_eager") if f"{k}_gbps" in bench[0]}
        if "naive_compiled_gbps" in bench[0]:
            series["naive_compiled"] = [b["naive_compiled_gbps"] for b in bench]
        plot_lines(args.cols, series, xlabel="row length (columns)", ylabel="effective GB/s (read+write)",
                   title=f"softmax, {args.rows} rows, {args.dtype}, {torch.cuda.get_device_name(0)}",
                   path=run.path("softmax_bandwidth.png"), logx=True)
        summary = {"device": torch.cuda.get_device_name(0), "correctness_ok": not bad, "bench": bench}
        run.save_summary(summary)
        if not args.smoke:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / "triton_softmax.json").write_text(json.dumps(summary, indent=2))
            import shutil

            shutil.copy(run.path("softmax_bandwidth.png"), args.report_dir / "triton_softmax_bandwidth.png")
        print(json.dumps({"run_dir": str(run.dir), "correctness_ok": not bad}, indent=2))
        for b in bench:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items() if not k.endswith("bytes_moved")})


if __name__ == "__main__":
    main()
