"""L09.5 — Agent task success with and without claim verification.

Tasks: a sample of rag-mini-wikipedia questions (needs search) and generated arithmetic questions
(needs the calculator, answer known exactly). Success = F1 >= 0.5 for QA, exact number for
arithmetic. The comparison reports what verification costs (extra calls, extra latency) and buys
(fewer unsupported claims, success change).

Run:  python applications/agent/evaluate.py --smoke
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from applications.agent.agent import Agent, Verifier  # noqa: E402
from applications.agent.tools import SearchTool, calculator, lookup_date  # noqa: E402
from applications.rag.evaluate import bootstrap_ci, exact_match, f1_score, load_qa  # noqa: E402
from applications.rag.index import DEFAULT_INDEX_DIR  # noqa: E402
from applications.trace import Tracer  # noqa: E402
from llm_systems_zoo import Run  # noqa: E402
from llm_systems_zoo.plotting import plot_bars  # noqa: E402
from llm_systems_zoo.registry import DATASETS, MODELS  # noqa: E402


def arithmetic_tasks(n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a, b, c = rng.randint(12, 999), rng.randint(2, 99), rng.randint(1, 999)
        expr = f"{a} * {b} + {c}"
        out.append({"id": f"arith{i}", "kind": "arith", "question": f"What is {expr}? Use the calculator.", "answer": str(eval(expr))})  # noqa: S307
    return out


def numeric_match(pred: str, gold: str) -> float:
    nums = re.findall(r"-?\d+(?:\.\d+)?", pred.replace(",", ""))
    return float(gold in nums)


def yes_no_match(pred: str, gold: str) -> float | None:
    """For yes/no golds, success is decided by the first word of the answer (None if not yes/no)."""
    g = gold.strip().lower().rstrip(".")
    if g not in ("yes", "no"):
        return None
    first = re.findall(r"[a-z]+", pred.lower())
    return float(bool(first) and first[0] == g)


def score(task: dict[str, Any], pred: str) -> dict[str, float]:
    if task["kind"] == "arith":
        s = numeric_match(pred, task["answer"])
        return {"success": s, "em": s, "f1": s}
    f1 = f1_score(pred, task["answer"])
    yn = yes_no_match(pred, task["answer"])
    success = yn if yn is not None else float(f1 >= 0.5)
    return {"success": success, "em": exact_match(pred, task["answer"]), "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ap.add_argument("--n-qa", type=int, default=40)
    ap.add_argument("--n-arith", type=int, default=10)
    ap.add_argument("--generator", choices=["hf", "endpoint"], default="hf")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--native-tools", action="store_true")
    ap.add_argument("--model", default=MODELS["small"].hf_id)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--rerank", action="store_true", help="rerank search results inside the tool")
    ap.add_argument("--verify-threshold", type=float, default=0.0, help="cross-encoder logit threshold")
    ap.add_argument("--verify-modes", nargs="+", default=["off", "flag", "remove"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_qa, args.n_arith, args.model, args.max_steps = 4, 2, MODELS["tiny"].hf_id, 2

    from applications.rag.pipeline import Reranker, Retriever, make_generator

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with Run("L09_app_to_infra", "agent_eval", cfg, model={"generator": args.model, "rerank": MODELS["rerank"].hf_id}, workload={"dataset": DATASETS["rag_qa"].hf_id, "n_qa": args.n_qa, "n_arith": args.n_arith}) as run:
        tasks = [{**q, "kind": "qa"} for q in load_qa(args.n_qa, seed=1)] + arithmetic_tasks(args.n_arith)
        retriever = Retriever(args.index_dir, device=args.device)
        reranker = Reranker(device=args.device)
        generator = make_generator(args.generator, args.model, args.endpoint, args.max_new_tokens, args.device)
        search = SearchTool(retriever, reranker if args.rerank else None)
        tools = {"search_passages": search, "calculator": calculator, "lookup_date": lambda: lookup_date("2026-10-29")}
        tracer = Tracer()
        results = []
        for mode in args.verify_modes:
            verifier = None if mode == "off" else Verifier(reranker.score, args.verify_threshold)
            agent = Agent(generator, tools, tracer, args.max_steps, verifier, mode, args.native_tools, evidence_source=search, max_new_tokens=args.max_new_tokens)
            rows = []
            for t in tasks:
                tracer.reset()
                out = agent.run(t["question"])
                sc = score(t, out["answer"])
                unsupported = sum(1 for v in (out["verification"] or []) if not v["supported"])
                rows.append({"id": t["id"], "kind": t["kind"], "question": t["question"], "gold": t["answer"], "pred": out["answer"], "raw": out["raw_answer"], **sc, "n_model_calls": out["n_model_calls"], "n_tool_calls": out["n_tool_calls"], "prompt_tokens": out["prompt_tokens"], "output_tokens": out["output_tokens"], "n_unsupported": unsupported, "parse_failed": any(s.get("parse_failed") for s in out["steps"]), **{f"{k}_s": v for k, v in out["latency_s"].items()}})
            run.log_records(rows, filename=f"tasks_{mode}.jsonl")
            tracer.save(run.path(f"last_trace_{mode}.json"))
            succ = [r["success"] for r in rows]
            lo, hi = bootstrap_ci(succ)

            def m(k: str, kind: str | None = None, _rows=rows) -> float:
                return float(np.mean([r[k] for r in _rows if kind is None or r["kind"] == kind]))

            res = {"verify_mode": mode, "n": len(rows), "success": m("success"), "success_ci95": [lo, hi], "success_qa": m("success", "qa"), "success_arith": m("success", "arith"), "f1_qa": m("f1", "qa"), "model_calls_mean": m("n_model_calls"), "tool_calls_mean": m("n_tool_calls"), "prompt_tokens_mean": m("prompt_tokens"), "unsupported_claims_mean": m("n_unsupported"), "parse_fail_rate": m("parse_failed"), "latency_s": {k: m(f"{k}_s") for k in ("wall", "model", "tool", "verify")}}
            results.append(res)
            lat = res["latency_s"]
            print(f"verify={mode:<6} success={res['success']:.2f} [{lo:.2f},{hi:.2f}] qa={res['success_qa']:.2f} arith={res['success_arith']:.2f} calls(model/tool)={res['model_calls_mean']:.1f}/{res['tool_calls_mean']:.1f} unsupported={res['unsupported_claims_mean']:.2f} | wall={lat['wall']:.2f}s model={lat['model']:.2f} tool={lat['tool']:.2f} verify={lat['verify']:.2f}", flush=True)
        labels, vals = [], []
        for r in results:
            for k in ("model", "tool", "verify"):
                labels.append(f"{r['verify_mode']}\n{k}")
                vals.append(r["latency_s"][k])
        plot_bars(labels, vals, ylabel="mean seconds per task", title="agent latency decomposition", path=run.path("latency_breakdown.png"))
        run.save_summary({"results": results, "device": str(getattr(generator, "device", "endpoint"))})
        print("run dir:", run.dir)


if __name__ == "__main__":
    main()
