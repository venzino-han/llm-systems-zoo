"""Eager vs torch.compile vs CUDA graphs for autoregressive decoding.

Part A (transformers): the same decode loop run eager, then compiled with each `torch.compile`
mode, with a *static* KV cache so shapes do not change between steps (dynamic shapes are the
first thing that breaks compiled decoding). The script separates

* warm-up cost: the first generate() call (graph capture + Inductor compile + autotune),
* recompilations: `torch._dynamo` counters before/after, with the reason if any,
* steady state: ms per generated token after warm-up, at several batch sizes.

Known failure (torch 2.11.0 + transformers 5.7.0): `mode="reduce-overhead"` with the static cache
logs "skipping cudagraphs due to mutated inputs" (the cache's in-place `cumulative_length.add_`)
and then **segfaults (exit 139)** during the first generate. Run modes one per process when
sweeping (`--modes reduce-overhead` alone) so a crash does not take the other rows with it;
`partial_rows.json` in the run directory keeps whatever finished.

Part B (vLLM): `enforce_eager=True` vs the default CUDA-graph decode at batch 1 and 8. vLLM
captures CUDA graphs for the decode step, which removes per-kernel launch overhead; that is the
whole gain at small batch and nothing at large batch, which is the point of measuring both.

    .venv/bin/python inference/compilation/bench_torch_compile.py --smoke
    .venv/bin/python inference/compilation/bench_torch_compile.py --part hf --modes eager default reduce-overhead
    .venv/bin/python inference/compilation/bench_torch_compile.py --part vllm --gpu-util 0.10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from inference.quantization.common import chat_prompts, load_prompts, vllm_env, wait_for_free_memory
from llm_systems_zoo import Run
from llm_systems_zoo.registry import MODELS


# ----------------------------------------------------------------------------- dynamo helpers
def dynamo_stats() -> dict:
    """Snapshot of torch._dynamo counters that matter for decode loops."""
    import torch._dynamo.utils as du

    c = du.counters
    return {
        "unique_graphs": int(c["stats"].get("unique_graphs", 0)),
        "calls_captured": int(c["stats"].get("calls_captured", 0)),
        "graph_breaks": int(sum(c["graph_break"].values())) if "graph_break" in c else 0,
        "recompile_reasons": {k: int(v) for k, v in c.get("recompiles", {}).items()} if "recompiles" in c else {},
    }


def recompiles_between(before: dict, after: dict) -> int:
    """A new unique graph after the first compile means a recompilation happened."""
    return max(after["unique_graphs"] - before["unique_graphs"], 0)


# ----------------------------------------------------------------------------- part A: HF
class RecompileLog(logging.Handler):
    """Collects torch._dynamo recompile reasons (enabled through torch._logging.set_logs)."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.reasons: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Recompiling" in msg or "triggered by" in msg or "guard" in msg.lower():
            self.reasons.append(msg.strip()[:300])


def hf_decode_bench(model_id: str, modes: list[str], batch_sizes: list[int], prompt_len: int, new_tokens: int, run: Run) -> list[dict]:
    import torch
    import torch._dynamo
    import torch._logging
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch._logging.set_logs(recompiles=True)
    rlog = RecompileLog()
    logging.getLogger("torch._dynamo").addHandler(rlog)

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = chat_prompts(tok, load_prompts(max(batch_sizes), min_chars=200))
    rows = []
    for mode in modes:
        torch._dynamo.reset()
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
        if mode != "eager":
            model.generation_config.cache_implementation = "static"
            model.forward = torch.compile(model.forward, mode=mode, fullgraph=True)
        for bs in batch_sizes:
            enc = tok(prompts[:bs], return_tensors="pt", padding=True, truncation=True, max_length=prompt_len).to("cuda")
            gen = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            before = dynamo_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(**enc, **gen)  # first call: compile + capture (+ autotune)
            torch.cuda.synchronize()
            warmup_s = time.perf_counter() - t0
            mid = dynamo_stats()
            t0 = time.perf_counter()
            model.generate(**enc, **gen)  # second call: catches shape-driven recompiles
            torch.cuda.synchronize()
            second_s = time.perf_counter() - t0
            after = dynamo_stats()
            times = []
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.generate(**enc, **gen)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            steady = min(times)
            row = {"part": "hf", "mode": mode, "batch": bs, "prompt_tokens": int(enc.input_ids.shape[1]), "new_tokens": new_tokens,
                   "first_call_s": warmup_s, "second_call_s": second_s, "steady_s": steady, "steady_ms_per_step": 1000 * steady / new_tokens,
                   "steady_output_tokens_per_s": bs * new_tokens / steady, "unique_graphs_before": before["unique_graphs"], "unique_graphs_after_first": mid["unique_graphs"],
                   "recompiles_on_second_call": recompiles_between(mid, after), "graph_breaks": after["graph_breaks"],
                   "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "recompile_reasons": rlog.reasons[-5:]}
            rlog.reasons.clear()
            run.log_metric(**{k: v for k, v in row.items() if k != "recompile_reasons"})
            rows.append(row)
            run.save_json("partial_rows.json", rows)  # survives a hard crash of a later mode
            print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
        torch._dynamo.reset()
    return rows


# ----------------------------------------------------------------------------- part B: vLLM
def vllm_one(model_id: str, enforce_eager: bool, batch_sizes: list[int], new_tokens: int, gpu_util: float, max_model_len: int, kv_cache_bytes: int) -> dict:
    """One vLLM engine (eager or CUDA graphs) in this process; main runs it in a subprocess."""
    vllm_env()
    import torch
    from vllm import SamplingParams

    from inference.quantization.common import VllmLogCapture, build_llm

    cap = VllmLogCapture().install()
    t0 = time.perf_counter()
    llm = build_llm(dict(model=model_id, enforce_eager=enforce_eager, gpu_memory_utilization=gpu_util, max_model_len=max_model_len, seed=0,
                         kv_cache_memory_bytes=kv_cache_bytes or None, max_num_seqs=64))
    init_s = time.perf_counter() - t0
    tok = llm.get_tokenizer()
    prompts = chat_prompts(tok, load_prompts(max(batch_sizes), min_chars=200))
    sp = SamplingParams(max_tokens=new_tokens, min_tokens=new_tokens, temperature=0.0, ignore_eos=True)
    llm.generate(prompts[:1], SamplingParams(max_tokens=4, temperature=0.0), use_tqdm=False)
    rows = []
    for bs in batch_sizes:
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = llm.generate(prompts[:bs], sp, use_tqdm=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        n_out = sum(len(o.outputs[0].token_ids) for o in out)
        rows.append({"part": "vllm", "mode": "enforce_eager" if enforce_eager else "cuda_graphs", "batch": bs, "new_tokens": new_tokens,
                     "steady_s": min(times), "steady_ms_per_step": 1000 * min(times) / new_tokens, "steady_output_tokens_per_s": n_out / min(times),
                     "engine_init_s": init_s, "cuda_graph_capture_s": cap.values.get("cuda_graph_capture_s"), "cuda_graph_memory_gib": cap.values.get("cuda_graph_memory_gib")})
    cap.remove()
    return {"status": "ok", "rows": rows, "device": torch.cuda.get_device_name(0)}


def vllm_bench(model_id: str, batch_sizes: list[int], new_tokens: int, gpu_util: float, max_model_len: int, run: Run, kv_cache_bytes: int = 0, min_free_gib: float = 0.0) -> list[dict]:
    from inference.quantization.common import run_in_subprocess

    rows = []
    for mode in ("eager", "graphs"):
        res = run_in_subprocess("inference/compilation/bench_torch_compile.py",
                                ["--one-vllm", mode, "--model", model_id, "--batch-sizes", *map(str, batch_sizes), "--new-tokens", str(new_tokens),
                                 "--gpu-util", str(gpu_util), "--max-model-len", str(max_model_len), "--kv-cache-bytes", str(kv_cache_bytes),
                                 "--min-free-gib", str(min_free_gib)], run.path(f"result_vllm_{mode}.json"))
        if res.get("status") != "ok":
            row = {"part": "vllm", "mode": "enforce_eager" if mode == "eager" else "cuda_graphs", "error": res.get("error")}
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        for row in res["rows"]:
            run.log_metric(**row)
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODELS["small"].hf_id)
    p.add_argument("--part", choices=["hf", "vllm", "both"], default="both")
    p.add_argument("--modes", nargs="+", default=["eager", "default", "reduce-overhead", "max-autotune"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--gpu-util", type=float, default=0.10)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--kv-cache-bytes", type=int, default=2 * 2**30, help="fixed vLLM KV budget in bytes (0 = size from --gpu-util)")
    p.add_argument("--min-free-gib", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--report-dir", type=Path, default=Path("reports/L08_inference_optimization/results"))
    p.add_argument("--one-vllm", choices=["eager", "graphs"], default=None, help="(internal) one vLLM engine in this process, write --out")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.smoke:
        args.modes, args.batch_sizes, args.new_tokens, args.part = ["eager", "default"], [1], 32, "hf"
    wait_for_free_memory(args.min_free_gib)
    if args.one_vllm:
        from llm_systems_zoo.io import dump_json

        try:
            res = vllm_one(args.model, args.one_vllm == "eager", args.batch_sizes, args.new_tokens, args.gpu_util, args.max_model_len, args.kv_cache_bytes)
        except Exception as e:
            res = {"status": "failed", "error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}
        dump_json(res, args.out)
        return
    import torch

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("one_vllm", "out")}
    with Run("L08_inference_optimization", "torch_compile", cfg, model={"id": args.model, "dtype": "bfloat16"},
             workload={"prompts": "ultrachat_200k first user turns, chat template", "batch_sizes": args.batch_sizes, "new_tokens": args.new_tokens, "decoding": "greedy"},
             execution={"label": "pilot, shared GPU", "repeats": 3, "statistic": "min of 3 after warm-up"}) as run:
        rows = []
        if args.part in ("hf", "both"):
            rows += hf_decode_bench(args.model, args.modes, args.batch_sizes, args.prompt_len, args.new_tokens, run)
        if args.part in ("vllm", "both"):
            rows += vllm_bench(args.model, args.batch_sizes, args.new_tokens, args.gpu_util, args.max_model_len, run, args.kv_cache_bytes, args.min_free_gib)
        summary = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": rows, "label": "pilot, shared GPU"}
        run.save_summary(summary)
        if not args.smoke:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / "torch_compile.json").write_text(json.dumps(summary, indent=2))
        print("run_dir", run.dir)


if __name__ == "__main__":
    main()
