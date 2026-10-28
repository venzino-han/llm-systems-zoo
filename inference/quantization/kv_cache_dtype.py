"""KV-cache quantisation in vLLM: `kv_cache_dtype=auto` (bf16) vs `fp8` (e5m2 / e4m3).

Halving the KV-cache element size doubles the number of tokens the same memory budget can hold,
which is the lever for long contexts and high concurrency. The cost is a dequantise step in the
attention kernel and a small accuracy change. This script measures, with a *fixed* KV memory
budget (same `gpu_memory_utilization` / `kv_cache_memory_bytes`):

* KV capacity in tokens and vLLM's "maximum concurrency" estimate (from the engine logs),
* decode throughput for a batch large enough to use that capacity,
* perplexity on wikitext-2 windows via prompt logprobs, so the accuracy cost is visible.

    .venv/bin/python inference/quantization/kv_cache_dtype.py --smoke
    .venv/bin/python inference/quantization/kv_cache_dtype.py --dtypes auto fp8 --batch 64
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from inference.quantization.common import (
    VllmLogCapture,
    build_llm,
    chat_prompts,
    load_prompts,
    run_in_subprocess,
    vllm_env,
    wait_for_free_memory,
    wikitext_token_ids,
)
from inference.quantization.eval_perplexity import window_bounds
from llm_systems_zoo import Run
from llm_systems_zoo.io import dump_json
from llm_systems_zoo.registry import MODELS


def run_one(args) -> dict:
    """One kv_cache_dtype in this process (child of main via --one)."""
    vllm_env()
    import torch
    from vllm import SamplingParams

    kv_dtype = args.one
    cap = VllmLogCapture().install()
    kwargs = dict(model=args.model, kv_cache_dtype=kv_dtype, gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
                  seed=0, disable_log_stats=False, max_num_seqs=args.max_num_seqs)
    if args.kv_cache_bytes:
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_bytes
    llm = build_llm(kwargs)
    tok = llm.get_tokenizer()
    ids = wikitext_token_ids(tok, args.ppl_tokens)
    win = min(1024, args.max_model_len - 1)
    windows = window_bounds(len(ids), win, win)
    outs = llm.generate([{"prompt_token_ids": ids[b:e].tolist()} for b, e, _ in windows],
                        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0), use_tqdm=False)
    nll, n = 0.0, 0
    for (b, e, first), out in zip(windows, outs):
        toks = ids[b:e].tolist()
        for j in range(first - b, e - b):
            nll -= out.prompt_logprobs[j][toks[j]].logprob
            n += 1
    prompts = chat_prompts(tok, load_prompts(args.batch))
    sp = SamplingParams(max_tokens=args.new_tokens, min_tokens=args.new_tokens, temperature=0.0, ignore_eos=True)
    llm.generate(prompts[:1], SamplingParams(max_tokens=4, temperature=0.0), use_tqdm=False)
    times = []
    for _ in range(2):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    n_out = sum(len(r.outputs[0].token_ids) for r in res)
    cap.remove()
    return {"status": "ok", "kv_cache_dtype": kv_dtype, "kv_cache_tokens": cap.values.get("kv_cache_tokens"), "max_concurrency_x": cap.values.get("max_concurrency_x"),
            "ppl": math.exp(nll / n), "ppl_tokens": n, "batch": args.batch, "output_tokens_per_s": n_out / min(times), "wall_s": min(times),
            "ms_per_step": 1000 * min(times) / args.new_tokens, "device": torch.cuda.get_device_name(0)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODELS["small"].hf_id)
    p.add_argument("--dtypes", nargs="+", default=["auto", "fp8"])
    p.add_argument("--batch", type=int, default=64, help="concurrent prompts for the throughput test")
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--gpu-util", type=float, default=0.10)
    p.add_argument("--kv-cache-bytes", type=int, default=2 * 2**30, help="fixed KV budget in bytes (0 = size from --gpu-util)")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--ppl-tokens", type=int, default=65536)
    p.add_argument("--min-free-gib", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--report-dir", type=Path, default=Path("reports/L08_inference_optimization/results"))
    p.add_argument("--one", default=None, help="(internal) run one kv dtype in this process and write --out")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.smoke:
        args.batch, args.new_tokens, args.ppl_tokens = 8, 32, 8192
    if args.one:
        wait_for_free_memory(args.min_free_gib)
        try:
            res = run_one(args)
        except Exception as e:
            res = {"status": "failed", "kv_cache_dtype": args.one, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}
        dump_json(res, args.out)
        return

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("one", "out")}
    with Run("L08_inference_optimization", "kv_cache_dtype", cfg, model={"id": args.model},
             workload={"throughput_prompts": "ultrachat_200k", "batch": args.batch, "new_tokens": args.new_tokens, "ppl": "wikitext-2 test, 1024-token windows"},
             execution={"label": "pilot, shared GPU", "isolation": "one subprocess per dtype"}) as run:
        passthrough = ["--model", args.model, "--batch", str(args.batch), "--new-tokens", str(args.new_tokens), "--gpu-util", str(args.gpu_util),
                       "--kv-cache-bytes", str(args.kv_cache_bytes), "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_num_seqs),
                       "--ppl-tokens", str(args.ppl_tokens), "--min-free-gib", str(args.min_free_gib)]
        rows = []
        for kv_dtype in args.dtypes:
            row = run_in_subprocess("inference/quantization/kv_cache_dtype.py", ["--one", kv_dtype, *passthrough], run.path(f"result_{kv_dtype}.json"))
            row.setdefault("kv_cache_dtype", kv_dtype)
            if row.get("status") == "ok":
                run.log_metric(**{k: v for k, v in row.items() if k != "device"})
            rows.append(row)
            print(json.dumps(row), flush=True)
        summary = {"device": next((r.get("device") for r in rows if r.get("device")), None), "model": args.model, "rows": rows, "label": "pilot, shared GPU"}
        run.save_summary(summary)
        if not args.smoke:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / "kv_cache_dtype.json").write_text(json.dumps(summary, indent=2, default=str))
        print("run_dir", run.dir)


if __name__ == "__main__":
    main()
