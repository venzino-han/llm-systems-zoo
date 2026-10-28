"""L02 — does automatic prefix caching help, and when? Real-engine measurement with vLLM.

Workload: N user prompts that all start with the same system prompt of P tokens. For each
engine configuration (prefix caching off / on) we measure, with `max_tokens=1` so the number is
pure prefill (a TTFT proxy without network/queueing effects):

    cold        first time the prefix is seen
    warm_same   the very same prompts again (upper bound for cache reuse)
    warm_new    new user suffixes behind the same system prompt (the realistic case)

and then full generation throughput with the cache warm. vLLM reports `num_cached_tokens` per
request and prefix-cache counters through `LLM.get_metrics()`, so the hit rate is read from the
engine, not inferred. The engine runs in its own process, so peak memory is taken from the
KV-cache size the engine reports at start-up rather than from torch in this process.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv/bin/python inference/kv_cache/bench_prefix_caching.py --smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from typing import Any

from llm_systems_zoo import Run
from llm_systems_zoo.plotting import plot_bars

LAB = "L02_kv_cache"


def build_prompts(tokenizer, prefix_tokens: int, n_prompts: int, suffix_tokens: int, seed: int) -> tuple[str, list[str], list[str]]:
    """A system prompt of exactly `prefix_tokens` tokens plus two disjoint sets of user prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True)
        users = []
        for ex in ds:
            text = ex["prompt"].strip().replace("\n", " ")
            if len(text) > 40:
                users.append(text)
            if len(users) >= 2 * n_prompts:
                break
        filler = " ".join(users)
    except Exception as e:  # offline fallback: synthetic but still natural-language-like text
        print(f"[warn] dataset unavailable ({e}); using synthetic prompts")
        base = "You are a careful assistant. Explain the trade-offs of paged attention and prefix caching in detail. "
        users = [f"Question {i}: how does block size {8 + i} interact with sharing? " * 3 for i in range(2 * n_prompts)]
        filler = base * 400
    ids = tokenizer(filler, add_special_tokens=False)["input_ids"][:prefix_tokens]
    prefix = tokenizer.decode(ids)
    # re-tokenise to make sure decode/encode round-trips to the requested length (±2 tokens)
    prefix = tokenizer.decode(tokenizer(prefix, add_special_tokens=False)["input_ids"][:prefix_tokens])
    trunc = [tokenizer.decode(tokenizer(u, add_special_tokens=False)["input_ids"][:suffix_tokens]) for u in users]
    return prefix, trunc[:n_prompts], trunc[n_prompts : 2 * n_prompts]


def prefix_cache_counters(llm) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        for m in llm.get_metrics():
            if "prefix_cache" in m.name and hasattr(m, "value"):
                out[m.name] = float(m.value)
    except Exception as e:
        out["error"] = str(e)[:100]
    return out


def timed_generate(llm, prompts: list[str], sp) -> tuple[float, list, int]:
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    cached = sum(getattr(o, "num_cached_tokens", 0) or 0 for o in outs)
    return dt, outs, cached


def run_condition(model: str, caching: bool, prefix_lens: list[int], n_prompts: int, suffix_tokens: int, max_tokens: int,
                  gpu_util: float, max_model_len: int, enforce_eager: bool, seed: int, run: Run) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, enable_prefix_caching=caching, gpu_memory_utilization=gpu_util, max_model_len=max_model_len,
              enforce_eager=enforce_eager, seed=seed, disable_log_stats=False)
    tok = llm.get_tokenizer()
    sp1 = SamplingParams(max_tokens=1, temperature=0)
    spg = SamplingParams(max_tokens=max_tokens, temperature=0, ignore_eos=True)
    rows = []
    # warm-up: an unrelated prompt so that engine start-up / kernel warm-up is not billed to "cold"
    llm.generate(["Warm-up prompt: list three prime numbers."] * 4, sp1, use_tqdm=False)
    for plen in prefix_lens:
        prefix, users_a, users_b = build_prompts(tok, plen, n_prompts, suffix_tokens, seed)
        prompts_a = [prefix + "\n\nUser: " + u for u in users_a]
        prompts_b = [prefix + "\n\nUser: " + u for u in users_b]
        n_prompt_tokens = sum(len(tok(p)["input_ids"]) for p in prompts_a)
        llm.reset_prefix_cache() if hasattr(llm, "reset_prefix_cache") else None
        c0 = prefix_cache_counters(llm)
        t_cold, _, cached_cold = timed_generate(llm, prompts_a, sp1)
        t_same, _, cached_same = timed_generate(llm, prompts_a, sp1)
        t_new, _, cached_new = timed_generate(llm, prompts_b, sp1)
        t_gen, outs, cached_gen = timed_generate(llm, prompts_b, spg)
        gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        c1 = prefix_cache_counters(llm)
        row = {
            "prefix_caching": caching, "prefix_tokens": plen, "n_prompts": n_prompts, "suffix_tokens": suffix_tokens,
            "prompt_tokens_per_batch": n_prompt_tokens,
            "prefill_only": {
                "cold_s": t_cold, "warm_same_s": t_same, "warm_new_suffix_s": t_new,
                "cold_prompt_tokens_per_s": n_prompt_tokens / t_cold,
                "warm_new_prompt_tokens_per_s": n_prompt_tokens / t_new,
                "speedup_warm_new_vs_cold": t_cold / t_new,
                "cached_tokens": {"cold": cached_cold, "warm_same": cached_same, "warm_new": cached_new},
                "cached_fraction_warm_new": cached_new / n_prompt_tokens,
            },
            "generation_warm": {"wall_s": t_gen, "output_tokens": gen_tokens, "output_tokens_per_s": gen_tokens / t_gen,
                                "cached_tokens": cached_gen},
            "engine_prefix_cache_counters": {"before": c0, "after": c1},
        }
        rows.append(row)
        run.log_metric(prefix_caching=caching, prefix_tokens=plen, cold_s=t_cold, warm_new_s=t_new, gen_tok_s=row["generation_warm"]["output_tokens_per_s"])
        print(f"[caching={caching!s:<5} prefix={plen:<5}] prefill cold {t_cold:.3f}s | same {t_same:.3f}s | new-suffix {t_new:.3f}s "
              f"(cached {cached_new}/{n_prompt_tokens} tok) | gen {row['generation_warm']['output_tokens_per_s']:.0f} tok/s")
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(3)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="vLLM prefix caching on/off benchmark.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prefix-tokens", type=int, nargs="+", default=[128, 512, 1024])
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--suffix-tokens", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--no-enforce-eager", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.prefix_tokens, a.n_prompts, a.max_tokens = [512], 8, 16
    name = "prefix_caching_smoke" if a.smoke else "prefix_caching"
    with Run(LAB, name, vars(a), model={"id": a.model, "engine": "vllm", "gpu_memory_utilization": a.gpu_memory_utilization,
                                        "max_model_len": a.max_model_len, "enforce_eager": not a.no_enforce_eager},
             workload={"prefix_tokens": a.prefix_tokens, "n_prompts": a.n_prompts, "suffix_tokens": a.suffix_tokens,
                       "max_tokens": a.max_tokens, "source": "HuggingFaceH4/ultrachat_200k test_sft (prompts)"},
             execution={"label": "pilot, shared GPU", "phases": ["cold", "warm_same", "warm_new", "generation"]}) as run:
        rows: list[dict[str, Any]] = []
        for caching in (False, True):
            rows += run_condition(a.model, caching, a.prefix_tokens, a.n_prompts, a.suffix_tokens, a.max_tokens,
                                  a.gpu_memory_utilization, a.max_model_len, not a.no_enforce_eager, a.seed, run)
        run.save_summary({"label": "pilot, shared GPU", "rows": rows})
        labels = [f"{'on' if r['prefix_caching'] else 'off'}\nP={r['prefix_tokens']}" for r in rows]
        plot_bars(labels, [r["prefill_only"]["warm_new_suffix_s"] for r in rows], ylabel="prefill wall time, new suffixes (s)",
                  title="Prefix caching: warm prefill of a batch sharing a system prompt", path=run.path("prefill_warm_new.png"))
        plot_bars(labels, [r["generation_warm"]["output_tokens_per_s"] for r in rows], ylabel="output tokens / s",
                  title="Generation throughput (cache warm)", path=run.path("generation_throughput.png"))
        print(json.dumps([{k: r[k] for k in ("prefix_caching", "prefix_tokens")} | {"speedup": r["prefill_only"]["speedup_warm_new_vs_cold"],
                           "cached_frac": r["prefill_only"]["cached_fraction_warm_new"]} for r in rows], indent=1))
        print(f"run dir: {run.dir}")


if __name__ == "__main__":
    main()
