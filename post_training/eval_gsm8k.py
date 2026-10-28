"""GSM8K accuracy with a bootstrap confidence interval (L06 quality gate).

    python post_training/eval_gsm8k.py --model small --n 100
    python post_training/eval_gsm8k.py --model small --adapter runs/L06_post_training/<run>/adapter --n 100

Greedy decoding, chat template with the same system prompt the GRPO reward expects. Reports
n, accuracy, a 95% bootstrap CI, and the format-compliance rate, so that a reward improvement
can be separated from a *format* improvement.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from llm_systems_zoo import Run
from llm_systems_zoo.io import dump_json
from post_training.common import LAB, bootstrap_ci, gsm8k_rows, resolve_model_id
from post_training.rewards import extract_answer, has_answer_format


def generate_hf(model_id: str, adapter: str | None, prompts: list[list[dict]], max_new_tokens: int, batch_size: int, device: str) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32).to(device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    outs: list[str] = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i : i + batch_size], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outs.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True))
    return outs


def generate_vllm(model_id: str, adapter: str | None, prompts: list[list[dict]], max_new_tokens: int, gpu_util: float) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, gpu_memory_utilization=gpu_util, enable_lora=adapter is not None, max_model_len=1024)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    kw = {}
    if adapter:
        from vllm.lora.request import LoRARequest

        kw["lora_request"] = LoRARequest("adapter", 1, adapter)
    outs = llm.chat(prompts, sp, **kw)
    return [o.outputs[0].text for o in outs]


def score(completions: list[str], golds: list[str]) -> dict:
    correct = [1.0 if extract_answer(c) == g else 0.0 for c, g in zip(completions, golds)]
    fmt = [1.0 if has_answer_format(c) else 0.0 for c in completions]
    return {"accuracy": bootstrap_ci(correct), "format_rate": sum(fmt) / len(fmt) if fmt else 0.0,
            "mean_completion_chars": sum(len(c) for c in completions) / len(completions) if completions else 0.0}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="small")
    p.add_argument("--adapter", default=None, help="LoRA adapter directory to merge before eval")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--split", default="test")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    p.add_argument("--vllm-gpu-util", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default=None)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out", default="reports/L06_post_training/results")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    if a.smoke:
        a.model, a.n, a.max_new_tokens = "tiny", 8, 48
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = resolve_model_id(a.model)
    rows = gsm8k_rows(a.split, a.n, a.seed)
    label = a.label or f"eval_{a.model}_{'adapter' if a.adapter else 'base'}"
    with Run(LAB, label, vars(a), root=a.runs_root, model={"id": model_id, "adapter": a.adapter},
             workload={"dataset": f"openai/gsm8k/main/{a.split}", "n": len(rows), "seed": a.seed, "decoding": "greedy"},
             execution={"backend": a.backend, "max_new_tokens": a.max_new_tokens}) as run:
        t0 = time.perf_counter()
        prompts = [r["prompt"] for r in rows]
        if a.backend == "vllm":
            comps = generate_vllm(model_id, a.adapter, prompts, a.max_new_tokens, a.vllm_gpu_util)
        else:
            comps = generate_hf(model_id, a.adapter, prompts, a.max_new_tokens, a.batch_size, device)
        wall = time.perf_counter() - t0
        res = score(comps, [r["answer"] for r in rows])
        res.update({"label": label, "model": model_id, "adapter": a.adapter, "n": len(rows), "wall_s": wall, "run_dir": str(run.dir)})
        run.log_records([{"question": r["question"], "gold": r["answer"], "completion": c, "pred": extract_answer(c)} for r, c in zip(rows, comps)])
        run.save_summary(res)
        if not a.smoke:
            dump_json(res, Path(a.out) / f"{label}.json")
        acc = res["accuracy"]
        print(f"[eval] {label}: n={acc['n']} accuracy={acc['mean']:.3f} (95% CI {acc['ci_low']:.3f}–{acc['ci_high']:.3f}) "
              f"format_rate={res['format_rate']:.2f} wall={wall:.1f}s")
        print("run dir:", run.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
