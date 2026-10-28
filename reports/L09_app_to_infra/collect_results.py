"""Copy the latest L09 runs into reports/L09_app_to_infra/results/ and print Markdown tables."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "L09_app_to_infra"
OUT = Path(__file__).resolve().parent / "results"


def latest(name: str) -> Path | None:
    cands = sorted(p for p in RUNS.glob(f"*_{name}_*") if (p / "summary.json").exists())
    return cands[-1] if cands else None


def copy_run(name: str, run: Path) -> dict:
    dst = OUT / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("summary.json", "manifest.json", "config.json"):
        shutil.copy(run / f, dst / f)
    for f in list(run.glob("*.png")) + list(run.glob("last_trace*.json")):
        shutil.copy(f, dst / f.name)
    (dst / "RUN_DIR.txt").write_text(str(run.relative_to(ROOT)) + "\n")
    return json.load(open(run / "summary.json"))


def md_table(headers: list[str], rows: list[list]) -> str:
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows])


def main() -> None:
    out = []
    run = latest("build_index")
    if run:
        s = copy_run("build_index", run)
        out.append(f"### Index build (`{run.relative_to(ROOT)}`)\n\n{s['n_passages']} passages, dim {s['dim']}, {s['embed_model']} on {s['device']}: model load {s['load_s']:.1f}s, embedding {s['embed_s']:.2f}s ({s['passages_per_s']:.0f} passages/s).\n")
    run = latest("rag_eval")
    if run:
        s = copy_run("rag_eval", run)
        cfg = json.load(open(run / "config.json"))
        rows = [[r["label"], r["n"], r["em"], f"[{r['em_ci95'][0]:.2f}, {r['em_ci95'][1]:.2f}]", r["f1"], r["recall_at_k"], r["recall_in_context"], r["prompt_tokens_mean"], r["latency_s"]["wall"] * 1e3, r["latency_s"]["retrieval"] * 1e3, r["latency_s"]["rerank"] * 1e3, r["latency_s"]["model"] * 1e3] for r in s["results"]]
        out.append(f"### RAG evaluation (`{run.relative_to(ROOT)}`, model={cfg['model']}, k={cfg['k']}, top_n={cfg['top_n']}, device={s['device']})\n\n" + md_table(["variant", "n", "EM", "EM 95% CI", "F1", f"recall@{cfg['k']}", "recall in context", "prompt tokens", "wall (ms)", "retrieval (ms)", "rerank (ms)", "model (ms)"], rows) + "\n")
    run = latest("agent_eval")
    if run:
        s = copy_run("agent_eval", run)
        cfg = json.load(open(run / "config.json"))
        rows = [[r["verify_mode"], r["n"], r["success"], f"[{r['success_ci95'][0]:.2f}, {r['success_ci95'][1]:.2f}]", r["success_qa"], r["success_arith"], r["f1_qa"], r["model_calls_mean"], r["tool_calls_mean"], r["prompt_tokens_mean"], r["unsupported_claims_mean"], r["parse_fail_rate"], r["latency_s"]["wall"], r["latency_s"]["model"], r["latency_s"]["tool"], r["latency_s"]["verify"]] for r in s["results"]]
        out.append(f"### Agent evaluation (`{run.relative_to(ROOT)}`, model={cfg['model']}, max_steps={cfg['max_steps']}, device={s['device']})\n\n" + md_table(["verify", "n", "success", "95% CI", "success QA", "success arith", "F1 QA", "model calls", "tool calls", "prompt tokens", "unsupported claims", "parse-fail rate", "wall (s)", "model (s)", "tool (s)", "verify (s)"], rows) + "\n")
    text = "\n".join(out)
    (OUT / "RESULTS.md").write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
