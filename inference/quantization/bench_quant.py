"""Orchestrates eval_perplexity.py across loading modes and builds the comparison table + figures.

Each (engine, mode) runs in its own subprocess so CUDA memory is fully released between modes
and a loader failure (missing kernel, unsupported GPU) is recorded instead of poisoning the rest.

    .venv/bin/python inference/quantization/bench_quant.py --smoke
    .venv/bin/python inference/quantization/bench_quant.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --hf-modes bf16 int8_bnb nf4_bnb --vllm-modes bf16 awq gptq gptq_int8 fp8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from llm_systems_zoo import Run
from llm_systems_zoo.io import load_json
from llm_systems_zoo.plotting import PALETTE, save_fig, setup_style
from llm_systems_zoo.registry import MODELS


def run_one(engine: str, mode: str, args, out: Path) -> dict:
    cmd = [sys.executable, "inference/quantization/eval_perplexity.py", "--engine", engine, "--load", mode, "--model", args.model,
           "--max-tokens", str(args.max_tokens), "--batch-sizes", *map(str, args.batch_sizes), "--max-new-tokens", str(args.max_new_tokens),
           "--gpu-util", str(args.gpu_util), "--max-model-len", str(args.max_model_len), "--min-free-gib", str(args.min_free_gib), "--out", str(out)]
    if args.smoke:
        cmd.append("--smoke")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    if out.exists():
        res = load_json(out)
    else:
        res = {"status": "failed", "engine": engine, "load": mode, "error": (proc.stderr.strip().splitlines() or ["no output"])[-1][:400]}
    res["returncode"] = proc.returncode
    return res


def table(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        row = {"engine": r.get("engine"), "mode": r.get("load"), "status": r.get("status"), "checkpoint": r.get("checkpoint")}
        if r.get("status") == "ok":
            row.update({"weights_gib": r.get("weights_gib"), "ppl": r["perplexity"]["ppl"], "scored_tokens": r["perplexity"]["scored_tokens"],
                        "prefill_tok_s": r["perplexity"]["prefill_tokens_per_s"]})
            for d in r["decode"]:
                row[f"decode_tok_s_b{d['batch']}"] = d["output_tokens_per_s"]
                row[f"ms_per_step_b{d['batch']}"] = d["ms_per_step"]
            row["kv_cache_tokens"] = r.get("kv_cache_tokens")
        else:
            row["error"] = r.get("error")
        rows.append(row)
    return rows


def figures(rows: list[dict], out_dir: Path, model: str) -> None:
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r["status"] == "ok" and r.get("weights_gib")]
    if not ok:
        return
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, r in enumerate(ok):
        label = f"{r['engine']}:{r['mode']}"
        axes[0].scatter(r["weights_gib"], r["ppl"], color=PALETTE[i % len(PALETTE)], label=label)
        axes[0].annotate(label, (r["weights_gib"], r["ppl"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        b1 = r.get("decode_tok_s_b1")
        if b1:
            axes[1].scatter(b1, r["ppl"], color=PALETTE[i % len(PALETTE)])
            axes[1].annotate(label, (b1, r["ppl"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set_xlabel("weights (GiB)")
    axes[0].set_ylabel("wikitext-2 perplexity")
    axes[0].set_title("quality vs memory")
    axes[1].set_xlabel("decode tokens/s, batch 1")
    axes[1].set_ylabel("wikitext-2 perplexity")
    axes[1].set_title("quality vs single-stream speed")
    fig.suptitle(model)
    save_fig(fig, out_dir / "quant_quality_tradeoff.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODELS["small"].hf_id)
    p.add_argument("--hf-modes", nargs="*", default=["bf16", "fp16", "int8_bnb", "nf4_bnb", "awq", "gptq"])
    p.add_argument("--vllm-modes", nargs="*", default=["bf16", "awq", "gptq", "gptq_int8", "fp8", "nf4_bnb"])
    p.add_argument("--max-tokens", type=int, default=65536, help="wikitext tokens for perplexity")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--gpu-util", type=float, default=0.10)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--min-free-gib", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--report-dir", type=Path, default=Path("reports/L08_inference_optimization/results"))
    args = p.parse_args()
    if args.smoke:
        args.hf_modes, args.vllm_modes = ["bf16"], []
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with Run("L08_inference_optimization", "bench_quant", cfg, model={"id": args.model}, execution={"label": "pilot, shared GPU", "isolation": "one subprocess per mode"}) as run:
        results = []
        for engine, modes in (("hf", args.hf_modes), ("vllm", args.vllm_modes)):
            for mode in modes:
                out = run.path(f"result_{engine}_{mode}.json")
                res = run_one(engine, mode, args, out)
                results.append(res)
                print(f"[{engine}:{mode}] status={res.get('status')} " + (f"ppl={res['perplexity']['ppl']:.3f} weights={res.get('weights_gib')}" if res.get("status") == "ok" else f"error={res.get('error')}"), flush=True)
        rows = table(results)
        run.save_json("table.json", rows)
        figures(rows, run.dir, args.model)
        summary = {"model": args.model, "rows": rows, "label": "pilot, shared GPU"}
        run.save_summary(summary)
        if not args.smoke:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            tag = args.model.split("/")[-1]
            (args.report_dir / f"quant_table_{tag}.json").write_text(json.dumps(summary, indent=2))
            if (run.dir / "quant_quality_tradeoff.png").exists():
                import shutil

                shutil.copy(run.dir / "quant_quality_tradeoff.png", args.report_dir / f"quant_quality_tradeoff_{tag}.png")
        print("run_dir", run.dir)
        print(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
