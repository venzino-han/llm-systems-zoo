"""Perplexity, memory and decode throughput for one model under one loading / quantisation mode.

Two engines:

* `--engine hf`   : transformers. Modes: bf16, fp16, int8_bnb (LLM.int8), nf4_bnb (QLoRA-style
                    NF4 + double quant), awq, gptq (need the autoawq / gptqmodel kernels; this
                    machine has neither, so the script records the loader error).
* `--engine vllm` : vLLM. Modes: bf16, fp16, awq, gptq, gptq_int8 (pre-quantised checkpoints),
                    fp8 (dynamic W8A8 if the GPU supports it, else vLLM's Marlin W8A16 fallback),
                    nf4_bnb (vLLM in-flight bitsandbytes 4-bit).

Perplexity is token-level on wikitext-2-raw-v1 (test) over non-overlapping windows of
`--max-length` tokens by default (`--stride` smaller than `--max-length` gives the sliding-window
estimate; both engines use the same windows so numbers are comparable across engines).

Examples (repo root):

    .venv/bin/python inference/quantization/eval_perplexity.py --engine hf --load nf4_bnb --smoke
    .venv/bin/python inference/quantization/eval_perplexity.py --engine vllm --load awq --gpu-util 0.10
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
    resolve_checkpoint,
    vllm_env,
    wait_for_free_memory,
    wikitext_token_ids,
)
from llm_systems_zoo import PhaseTimer, Run
from llm_systems_zoo.io import dump_json
from llm_systems_zoo.registry import MODELS

HF_MODES = ("bf16", "fp16", "int8_bnb", "nf4_bnb", "awq", "gptq")
VLLM_MODES = ("bf16", "fp16", "awq", "gptq", "gptq_int8", "fp8", "nf4_bnb")


# ----------------------------------------------------------------------------- perplexity math
def window_bounds(seq_len: int, max_length: int, stride: int) -> list[tuple[int, int, int]]:
    """(begin, end, first_scored_abs_index) for each window. Shared by both engines and the tests."""
    out = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        first_scored = max(prev_end, begin + 1)  # token 0 of a window is never predicted
        out.append((begin, end, first_scored))
        prev_end = end
        if end == seq_len:
            break
    return out


def nll_from_logits(logits, input_ids, first_scored_offset: int) -> tuple[float, int]:
    """Sum of -log p(token_j | tokens_<j) for j >= first_scored_offset (window-relative)."""
    import torch

    logits = logits[first_scored_offset - 1 : -1]
    targets = input_ids[first_scored_offset:]
    total = 0.0
    # Chunk the fp32 log-softmax: a 1024 x 152k vocab window would otherwise need ~0.6 GiB extra.
    for i in range(0, logits.shape[0], 256):
        total += float(torch.nn.functional.cross_entropy(logits[i : i + 256].float(), targets[i : i + 256], reduction="sum").item())
    return total, int(targets.numel())


# ----------------------------------------------------------------------------- HF engine
def load_hf(ckpt: str, load: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs: dict = {"device_map": {"": 0}}
    if load == "bf16":
        kwargs["dtype"] = torch.bfloat16
    elif load == "fp16":
        kwargs["dtype"] = torch.float16
    elif load == "int8_bnb":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["dtype"] = torch.float16
    elif load == "nf4_bnb":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        )
        kwargs["dtype"] = torch.bfloat16
    elif load in ("awq", "gptq"):
        kwargs["dtype"] = torch.float16  # the checkpoint's quantization_config selects the kernel
    else:
        raise ValueError(f"unsupported HF load mode {load}")
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, **kwargs).eval()
    return model, tok


def perplexity_hf(model, ids, max_length: int, stride: int) -> dict:
    import torch

    device = next(model.parameters()).device
    nll, n = 0.0, 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for begin, end, first_scored in window_bounds(len(ids), max_length, stride):
            window = ids[begin:end].to(device)
            logits = model(window[None]).logits[0]
            s, c = nll_from_logits(logits, window, first_scored - begin)
            nll += s
            n += c
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"ppl": math.exp(nll / n), "nll_per_token": nll / n, "scored_tokens": n, "eval_s": dt, "prefill_tokens_per_s": len(ids) / dt}


def decode_throughput_hf(model, tok, prompts: list[str], batch_sizes: list[int], max_new_tokens: int) -> list[dict]:
    import torch

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = []
    for bs in batch_sizes:
        batch = chat_prompts(tok, prompts[:bs])
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        gen = dict(max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        model.generate(**enc, **{**gen, "max_new_tokens": 4, "min_new_tokens": 4})  # warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**enc, **gen)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new_tokens = int((out[:, enc.input_ids.shape[1] :] != tok.pad_token_id).sum().item())
        rows.append({"batch": bs, "prompt_tokens": int(enc.input_ids.numel()), "output_tokens": new_tokens, "wall_s": dt,
                     "output_tokens_per_s": new_tokens / dt, "ms_per_step": 1000 * dt / max_new_tokens})
    return rows


def run_hf(args, run: Run) -> dict:
    import torch

    ckpt = resolve_checkpoint(args.model, args.load)
    timer = PhaseTimer(sync_cuda=True)
    torch.cuda.reset_peak_memory_stats()
    with timer.phase("load"):
        model, tok = load_hf(ckpt, args.load)
    weights_gib = model.get_memory_footprint() / 2**30
    run.update(model={"checkpoint": ckpt, "load": args.load, "dtype": str(next(model.parameters()).dtype), "weights_gib": weights_gib})
    ids = wikitext_token_ids(tok, args.max_tokens)
    with timer.phase("perplexity"):
        ppl = perplexity_hf(model, ids, args.max_length, args.stride)
    ppl_peak_gib = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    prompts = load_prompts(max(args.batch_sizes))
    with timer.phase("decode"):
        decode = decode_throughput_hf(model, tok, prompts, args.batch_sizes, args.max_new_tokens)
    decode_peak_gib = torch.cuda.max_memory_allocated() / 2**30
    return {"engine": "hf", "checkpoint": ckpt, "load": args.load, "weights_gib": weights_gib, "ppl_peak_gib": ppl_peak_gib,
            "decode_peak_gib": decode_peak_gib, "perplexity": ppl, "decode": decode, "phases": timer.summary(), "device": torch.cuda.get_device_name(0)}


# ----------------------------------------------------------------------------- vLLM engine
def run_vllm(args, run: Run) -> dict:
    vllm_env()
    import torch
    from vllm import SamplingParams

    ckpt = resolve_checkpoint(args.model, args.load)
    kwargs: dict = dict(model=ckpt, gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len, seed=0,
                        enforce_eager=args.enforce_eager, disable_log_stats=False, max_num_seqs=args.max_num_seqs)
    if args.kv_cache_bytes:
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_bytes
    if args.load in ("bf16", "fp16"):
        kwargs["dtype"] = "bfloat16" if args.load == "bf16" else "float16"
    elif args.load == "fp8":
        kwargs["quantization"] = "fp8"
    elif args.load == "nf4_bnb":
        kwargs["quantization"] = "bitsandbytes"
    # awq / gptq / gptq_int8: the checkpoint's quantization_config picks the kernel (awq_marlin, gptq_marlin...)
    cap = VllmLogCapture().install()
    timer = PhaseTimer(sync_cuda=True)
    with timer.phase("load"):
        llm = build_llm(kwargs)
    mem_after_load_gib = torch.cuda.memory_allocated() / 2**30
    tok = llm.get_tokenizer()
    run.update(model={"checkpoint": ckpt, "load": args.load, "engine_kwargs": {k: v for k, v in kwargs.items() if k != "model"}, **cap.values})

    ids = wikitext_token_ids(tok, args.max_tokens)
    max_len = min(args.max_length, args.max_model_len - 1)
    windows = window_bounds(len(ids), max_len, min(args.stride, max_len))
    prompts_tok = [{"prompt_token_ids": ids[b:e].tolist()} for b, e, _ in windows]
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    with timer.phase("perplexity"):
        outs = llm.generate(prompts_tok, sp, use_tqdm=False)
    nll, n = 0.0, 0
    for (b, e, first), out in zip(windows, outs):
        plp = out.prompt_logprobs  # list aligned with prompt tokens; entry 0 is None
        toks = ids[b:e].tolist()
        for j in range(first - b, e - b):
            nll -= plp[j][toks[j]].logprob
            n += 1
    ppl = {"ppl": math.exp(nll / n), "nll_per_token": nll / n, "scored_tokens": n, "eval_s": timer.totals["perplexity"],
           "prefill_tokens_per_s": len(ids) / timer.totals["perplexity"]}

    prompts = chat_prompts(tok, load_prompts(max(args.batch_sizes)))
    gen = SamplingParams(max_tokens=args.max_new_tokens, min_tokens=args.max_new_tokens, temperature=0.0, ignore_eos=True)
    llm.generate(prompts[:1], SamplingParams(max_tokens=4, temperature=0.0), use_tqdm=False)  # warm-up
    decode = []
    for bs in args.batch_sizes:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = llm.generate(prompts[:bs], gen, use_tqdm=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new_tokens = sum(len(r.outputs[0].token_ids) for r in res)
        decode.append({"batch": bs, "prompt_tokens": sum(len(r.prompt_token_ids) for r in res), "output_tokens": new_tokens,
                       "wall_s": dt, "output_tokens_per_s": new_tokens / dt, "ms_per_step": 1000 * dt / args.max_new_tokens})
    cap.remove()
    return {"engine": "vllm", "checkpoint": ckpt, "load": args.load, "weights_gib": cap.values.get("model_weights_gib"),
            "mem_after_load_gib": mem_after_load_gib, "kv_cache_tokens": cap.values.get("kv_cache_tokens"),
            "perplexity": ppl, "decode": decode, "phases": timer.summary(), "device": torch.cuda.get_device_name(0),
            "vllm_log": cap.values}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODELS["small"].hf_id)
    p.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    p.add_argument("--load", default="bf16", help=f"hf: {HF_MODES}; vllm: {VLLM_MODES}")
    p.add_argument("--max-length", type=int, default=1024, help="perplexity window length (tokens)")
    p.add_argument("--stride", type=int, default=1024, help="window stride; < max-length gives sliding-window ppl")
    p.add_argument("--max-tokens", type=int, default=0, help="limit wikitext tokens (0 = full test set, ~300k tokens)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--gpu-util", type=float, default=0.10, help="vLLM gpu_memory_utilization")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--kv-cache-bytes", type=int, default=2 * 2**30, help="vLLM kv_cache_memory_bytes; fixed KV budget that also skips the free-memory profiling race on shared GPUs (0 = size from --gpu-util)")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=256, help="vLLM max_num_seqs; lower it when memory is tight (sampler warm-up allocates per seq)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--min-free-gib", type=float, default=0.0, help="wait until this much GPU memory is free before loading (shared GPUs)")
    p.add_argument("--out", type=Path, default=None, help="also write the summary JSON here (used by bench_quant.py)")
    args = p.parse_args()
    wait_for_free_memory(args.min_free_gib)
    if args.smoke:
        args.max_tokens, args.batch_sizes, args.max_new_tokens = 8192, [1, 4], 32
    modes = HF_MODES if args.engine == "hf" else VLLM_MODES
    if args.load not in modes:
        raise SystemExit(f"--load {args.load} is not supported by engine {args.engine}; choose from {modes}")

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    name = f"ppl_{args.engine}_{args.load}_{args.model.split('/')[-1]}"
    workload = {"ppl_dataset": "wikitext/wikitext-2-raw-v1/test", "window": args.max_length, "stride": args.stride,
                "max_tokens": args.max_tokens, "decode_prompts": "HuggingFaceH4/ultrachat_200k test_sft (first user turn)",
                "batch_sizes": args.batch_sizes, "max_new_tokens": args.max_new_tokens, "decoding": "greedy, min_tokens=max_tokens"}
    with Run("L08_inference_optimization", name, cfg, workload=workload, execution={"shared_gpu": True, "label": "pilot, shared GPU"}) as run:
        try:
            summary = run_hf(args, run) if args.engine == "hf" else run_vllm(args, run)
            summary["status"] = "ok"
        except Exception as e:  # record loader/kernels failures as data, then re-raise
            summary = {"status": "failed", "engine": args.engine, "load": args.load, "checkpoint": resolve_checkpoint(args.model, args.load),
                       "error": f"{type(e).__name__}: {str(e).splitlines()[0][:400]}"}
            run.save_summary(summary)
            if args.out:
                dump_json(summary, args.out)
            raise
        summary["run_dir"] = str(run.dir)
        run.save_summary(summary)
        if args.out:
            dump_json(summary, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("phases", "vllm_log")}, indent=2, default=str))


if __name__ == "__main__":
    main()
