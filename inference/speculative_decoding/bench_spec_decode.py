"""Speculative decoding in vLLM: n-gram (prompt lookup) and draft-model proposers vs plain decode.

Measures, for each configuration and batch size:
* decode throughput (output tokens/s) and ms per step on a fixed prompt set, greedy,
* the engine's own acceptance statistics (`vllm:spec_decode_num_accepted_tokens`,
  `vllm:spec_decode_num_draft_tokens`, per-position counters) from `llm.get_metrics()`,
* whether the greedy output is identical to the non-speculative run (it should be: verification
  makes speculative decoding lossless under greedy sampling).

Speculation pays when the target step is memory-bound and the batch is small; as the batch grows
the verification forward stops being "free" and the draft overhead shows. The batch sweep is
there to find that crossover on this GPU.

    .venv/bin/python inference/speculative_decoding/bench_spec_decode.py --smoke
    .venv/bin/python inference/speculative_decoding/bench_spec_decode.py --methods none ngram draft --batch-sizes 1 4 16
"""

from __future__ import annotations

import argparse
import json
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
)
from llm_systems_zoo import Run
from llm_systems_zoo.io import dump_json
from llm_systems_zoo.registry import MODELS


def spec_config(method: str, draft: str, k: int) -> dict | None:
    if method == "none":
        return None
    if method == "ngram":
        return {"method": "ngram", "num_speculative_tokens": k, "prompt_lookup_max": 5, "prompt_lookup_min": 2}
    if method == "draft":
        return {"method": "draft_model", "model": draft, "num_speculative_tokens": k}
    raise ValueError(method)


def spec_metrics(llm) -> dict:
    """Pull acceptance counters out of vLLM's Prometheus-style metrics (V1 engine)."""
    out: dict = {}
    try:
        for m in llm.get_metrics():
            name = getattr(m, "name", "")
            if "spec_decode" in name:
                val = getattr(m, "value", None)
                if val is None and hasattr(m, "values"):
                    val = list(m.values)
                out[name] = val
    except Exception as e:  # metrics API differs across versions; keep the error as data
        out["error"] = f"{type(e).__name__}: {e}"
    acc, draft = out.get("vllm:spec_decode_num_accepted_tokens"), out.get("vllm:spec_decode_num_draft_tokens")
    if isinstance(acc, (int, float)) and isinstance(draft, (int, float)) and draft:
        out["acceptance_rate"] = acc / draft
    drafts = out.get("vllm:spec_decode_num_drafts")
    if isinstance(acc, (int, float)) and isinstance(drafts, (int, float)) and drafts:
        out["mean_accepted_per_step"] = acc / drafts
    return out


def run_one(args) -> dict:
    """One speculation method, all batch sizes, in this process (called via --one by main)."""
    vllm_env()
    import torch
    from vllm import SamplingParams

    method = args.one
    cap = VllmLogCapture().install()
    kwargs = dict(model=args.target, gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len, seed=0,
                  disable_log_stats=False, max_num_seqs=args.max_num_seqs)
    if args.kv_cache_bytes:
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_bytes
    sc = spec_config(method, args.draft, args.k)
    if sc:
        kwargs["speculative_config"] = sc
    t0 = time.perf_counter()
    llm = build_llm(kwargs)
    init_s = time.perf_counter() - t0
    tok = llm.get_tokenizer()
    prompts = chat_prompts(tok, load_prompts(max(args.batch_sizes), min_chars=200))
    sp = SamplingParams(max_tokens=args.new_tokens, min_tokens=args.new_tokens, temperature=0.0, ignore_eos=True)
    llm.generate(prompts[:1], SamplingParams(max_tokens=4, temperature=0.0), use_tqdm=False)
    rows = []
    for bs in args.batch_sizes:
        times, outs = [], None
        for _ in range(2):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = llm.generate(prompts[:bs], sp, use_tqdm=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        token_ids = [list(o.outputs[0].token_ids) for o in outs]
        n_out = sum(len(t) for t in token_ids)
        rows.append({"method": method, "k": args.k if method != "none" else 0, "batch": bs, "new_tokens": args.new_tokens, "wall_s": min(times),
                     "ms_per_step": 1000 * min(times) / args.new_tokens, "output_tokens_per_s": n_out / min(times), "engine_init_s": init_s,
                     "spec_metrics": spec_metrics(llm) if method != "none" else {}, "kv_cache_tokens": cap.values.get("kv_cache_tokens"),
                     "token_ids": token_ids})
    cap.remove()
    return {"status": "ok", "method": method, "rows": rows, "device": torch.cuda.get_device_name(0)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default=MODELS["medium"].hf_id)
    p.add_argument("--draft", default=MODELS["draft"].hf_id)
    p.add_argument("--methods", nargs="+", default=["none", "ngram", "draft"])
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--gpu-util", type=float, default=0.10)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--kv-cache-bytes", type=int, default=2 * 2**30, help="fixed KV budget in bytes (0 = size from --gpu-util)")
    p.add_argument("--min-free-gib", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--report-dir", type=Path, default=Path("reports/L08_inference_optimization/results"))
    p.add_argument("--one", default=None, help="(internal) run a single method in this process and write --out")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.smoke:
        args.target, args.methods, args.batch_sizes, args.new_tokens = MODELS["small"].hf_id, ["none", "ngram"], [1, 4], 32
    if args.one:  # child process: one engine, write JSON, exit
        wait_for_free_memory(args.min_free_gib)
        try:
            res = run_one(args)
        except Exception as e:
            res = {"status": "failed", "method": args.one, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}
        dump_json(res, args.out)
        return

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("one", "out")}
    with Run("L08_inference_optimization", "vllm_spec_decode", cfg, model={"target": args.target, "draft": args.draft, "k": args.k},
             workload={"prompts": "ultrachat_200k first user turns", "batch_sizes": args.batch_sizes, "new_tokens": args.new_tokens, "decoding": "greedy, ignore_eos"},
             execution={"label": "pilot, shared GPU", "repeats": 2, "statistic": "min", "isolation": "one subprocess per method"}) as run:
        rows, reference = [], {}
        passthrough = ["--target", args.target, "--draft", args.draft, "--k", str(args.k), "--batch-sizes", *map(str, args.batch_sizes),
                       "--new-tokens", str(args.new_tokens), "--gpu-util", str(args.gpu_util), "--max-model-len", str(args.max_model_len),
                       "--max-num-seqs", str(args.max_num_seqs), "--kv-cache-bytes", str(args.kv_cache_bytes), "--min-free-gib", str(args.min_free_gib)]
        for method in args.methods:
            res = run_in_subprocess("inference/speculative_decoding/bench_spec_decode.py", ["--one", method, *passthrough], run.path(f"result_{method}.json"))
            if res.get("status") != "ok":
                row = {"method": method, "error": res.get("error")}
                rows.append(row)
                print(json.dumps(row), flush=True)
                continue
            for row in res["rows"]:
                ids = row.pop("token_ids")
                if method == "none":
                    reference[row["batch"]] = ids
                    row["equal_to_plain_greedy"] = None
                else:
                    ref = reference.get(row["batch"])
                    row["equal_to_plain_greedy"] = sum(a == b for a, b in zip(ids, ref)) / len(ids) if ref else None
                run.log_metric(**{k: v for k, v in row.items() if k != "spec_metrics"})
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "spec_metrics"}), {k: v for k, v in row["spec_metrics"].items() if "per_pos" not in k}, flush=True)
        summary = {"device": rows and next((r.get("device") for r in rows if r.get("device")), None), "target": args.target, "draft": args.draft, "rows": rows, "label": "pilot, shared GPU"}
        run.save_summary(summary)
        if not args.smoke:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / "vllm_spec_decode.json").write_text(json.dumps(summary, indent=2, default=str))
        print("run_dir", run.dir)


if __name__ == "__main__":
    main()
