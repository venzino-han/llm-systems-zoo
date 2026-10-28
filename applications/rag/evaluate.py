"""L09.3 — Evaluate the RAG pipeline: answer quality, retrieval recall, and per-stage latency.

Metrics
* exact match (EM) and token F1 with SQuAD-style normalisation, with a bootstrap CI on EM
* recall@k: does any retrieved passage contain the gold answer string?
* latency breakdown per stage from the tracer (retrieval / rerank / model)

Run:  python applications/rag/evaluate.py --smoke
"""

from __future__ import annotations

import argparse
import collections
import re
import string
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from applications.rag.index import DEFAULT_INDEX_DIR  # noqa: E402
from applications.trace import Tracer  # noqa: E402
from llm_systems_zoo import Run  # noqa: E402
from llm_systems_zoo.plotting import plot_bars  # noqa: E402
from llm_systems_zoo.registry import DATASETS, MODELS  # noqa: E402


# ---- SQuAD-style scoring (pure, unit-tested) ----------------------------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def recall_at_k(gold: str, passages: Sequence[str]) -> float:
    g = normalize_answer(gold)
    return float(any(g and g in normalize_answer(p) for p in passages))


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, arr.size, replace=True).mean() for _ in range(n_boot)])
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


# ---- evaluation ------------------------------------------------------------------------------
def load_qa(limit: int, seed: int = 0) -> list[dict[str, Any]]:
    from datasets import load_dataset

    spec = DATASETS["rag_qa"]
    ds = load_dataset(spec.hf_id, spec.config, split=spec.split).shuffle(seed=seed)
    return [{"id": int(r["id"]), "question": str(r["question"]), "answer": str(r["answer"])} for r in ds.select(range(min(limit, len(ds))))]


def evaluate_pipeline(pipeline, qa: list[dict[str, Any]], tracer: Tracer, label: str) -> dict[str, Any]:
    per_q = []
    for ex in qa:
        tracer.reset()
        t0 = time.perf_counter()
        out = pipeline.answer(ex["question"])
        wall = time.perf_counter() - t0
        bd = tracer.breakdown()
        per_q.append({
            "id": ex["id"],
            "question": ex["question"],
            "gold": ex["answer"],
            "pred": out["answer"],
            "em": exact_match(out["answer"], ex["answer"]),
            "f1": f1_score(out["answer"], ex["answer"]),
            "recall_at_k": recall_at_k(ex["answer"], [c["passage"] for c in out["candidates"]]),
            "recall_in_context": recall_at_k(ex["answer"], [c["passage"] for c in out["context"]]),
            "prompt_tokens": out["prompt_tokens"],
            "output_tokens": out["output_tokens"],
            "wall_s": wall,
            "retrieval_s": bd["retrieval"]["total_s"],
            "rerank_s": bd["rerank"]["total_s"],
            "model_s": bd["model"]["total_s"],
        })
    ems = [q["em"] for q in per_q]
    lo, hi = bootstrap_ci(ems)
    mean = lambda k: float(np.mean([q[k] for q in per_q]))  # noqa: E731
    return {
        "label": label,
        "n": len(per_q),
        "em": mean("em"),
        "em_ci95": [lo, hi],
        "f1": mean("f1"),
        "recall_at_k": mean("recall_at_k"),
        "recall_in_context": mean("recall_in_context"),
        "prompt_tokens_mean": mean("prompt_tokens"),
        "latency_s": {"wall": mean("wall_s"), "retrieval": mean("retrieval_s"), "rerank": mean("rerank_s"), "model": mean("model_s")},
        "per_question": per_q,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--rerank", choices=["off", "on", "both"], default="both")
    ap.add_argument("--generator", choices=["hf", "endpoint"], default="hf")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--model", default=MODELS["small"].hf_id)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.model, args.k = 8, MODELS["tiny"].hf_id, 5

    from applications.rag.pipeline import RAGPipeline, Reranker, Retriever, make_generator

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with Run("L09_app_to_infra", "rag_eval", cfg, model={"generator": args.model, "embed": MODELS["embed"].hf_id, "rerank": MODELS["rerank"].hf_id}, workload={"dataset": DATASETS["rag_qa"].hf_id, "n": args.n}) as run:
        qa = load_qa(args.n)
        tracer = Tracer()
        retriever = Retriever(args.index_dir, device=args.device)
        generator = make_generator(args.generator, args.model, args.endpoint, args.max_new_tokens, args.device)
        reranker = Reranker(device=args.device) if args.rerank != "off" else None
        results = []
        variants = {"off": [False], "on": [True], "both": [False, True]}[args.rerank]
        for use_rr in variants:
            pipe = RAGPipeline(retriever, generator, reranker if use_rr else None, tracer, args.k, args.top_n)
            label = "rerank" if use_rr else "no_rerank"
            res = evaluate_pipeline(pipe, qa, tracer, label)
            results.append(res)
            run.log_records(res["per_question"], filename=f"per_question_{label}.jsonl")
            lat = res["latency_s"]
            print(f"{label:<10} n={res['n']} EM={res['em']:.3f} [{res['em_ci95'][0]:.3f},{res['em_ci95'][1]:.3f}] F1={res['f1']:.3f} recall@{args.k}={res['recall_at_k']:.3f} recall@ctx={res['recall_in_context']:.3f} | wall={lat['wall'] * 1e3:.0f}ms retr={lat['retrieval'] * 1e3:.0f} rerank={lat['rerank'] * 1e3:.0f} model={lat['model'] * 1e3:.0f}", flush=True)
        tracer.save(run.path("last_trace.json"))
        labels, vals = [], []
        for r in results:
            for stage in ("retrieval", "rerank", "model"):
                labels.append(f"{r['label']}\n{stage}")
                vals.append(r["latency_s"][stage] * 1e3)
        plot_bars(labels, vals, ylabel="mean latency per question (ms)", title=f"{args.model}", path=run.path("latency_breakdown.png"))
        run.save_summary({"results": [{k: v for k, v in r.items() if k != "per_question"} for r in results], "device": str(getattr(generator, "device", "endpoint"))})
        print("run dir:", run.dir)


if __name__ == "__main__":
    main()
