"""L09.4 — A minimal tool-calling agent with a verification step, fully traced.

Protocol (default): the model answers with exactly one JSON object,
    {"thought": "...", "action": "<tool name>", "args": {...}}      or
    {"thought": "...", "action": "final", "answer": "..."}
Native OpenAI tool calling is used instead when `native_tools=True` and the endpoint supports it.

Verification: every sentence of the final answer is treated as a claim and scored against the
evidence gathered during the run (retrieved passages via the cross-encoder reranker; tool
observations by substring match). Unsupported claims are flagged or removed. LangGraph / MCP are
optional integrations for the same loop; nothing here depends on them.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from applications.agent.tools import TOOL_SPECS, openai_tool_defs, render_tool_help  # noqa: E402
from applications.trace import Tracer  # noqa: E402

AGENT_SYSTEM_PROMPT = """You are a careful assistant that solves tasks step by step using tools.
Available tools:
{tools}

Respond with exactly ONE JSON object and nothing else, in one of two forms:
{{"thought": "<brief reasoning>", "action": "<tool name>", "args": {{<arguments>}}}}
{{"thought": "<brief reasoning>", "action": "final", "answer": "<short final answer>"}}
Use search_passages for factual questions and calculator for arithmetic. Keep the final answer as short as possible (a name, number, date or phrase)."""


# ---- parsing (pure, unit-tested) --------------------------------------------------------------
def parse_action(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object with an "action" key; tolerate code fences, prose, and a
    JSON object truncated by the token limit (salvaged with regexes, marked `salvaged=True`)."""
    text = re.sub(r"```(?:json)?", "", text)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "action" in obj:
            obj["action"] = str(obj["action"]).strip()
            obj.setdefault("args", {})
            if not isinstance(obj["args"], dict):
                obj["args"] = {}
            return obj
    act = re.search(r'"action"\s*:\s*"([^"]+)"', text)
    if not act:
        return None
    action = act.group(1).strip()
    if action == "final":
        ans = re.search(r'"answer"\s*:\s*"([^"]*)', text)
        return {"action": "final", "answer": ans.group(1).strip() if ans else "", "args": {}, "salvaged": True}
    args_blob = re.search(r'"args"\s*:\s*\{([^}]*)', text)
    args = dict(re.findall(r'"(\w+)"\s*:\s*"([^"]*)"', args_blob.group(1))) if args_blob else {}
    return {"action": action, "args": args, "salvaged": True}


def split_claims(answer: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [p.strip() for p in parts if len(p.strip()) > 1]


def decide_support(claim: str, passage_scores: list[float], tool_observations: list[str], threshold: float) -> dict[str, Any]:
    """A claim is supported if a passage scores >= threshold OR every number in it appears in a tool observation."""
    best = max(passage_scores) if passage_scores else None
    if best is not None and best >= threshold:
        return {"supported": True, "reason": "passage", "score": best}
    nums = re.findall(r"-?\d+(?:\.\d+)?", claim)
    if nums and tool_observations:
        obs = " ".join(tool_observations)
        if all(n in obs for n in nums):
            return {"supported": True, "reason": "tool_observation", "score": best}
    return {"supported": False, "reason": "no_evidence", "score": best}


# ---- agent ----------------------------------------------------------------------------------
class Verifier:
    def __init__(self, scorer: Callable[[str, list[str]], list[float]] | None, threshold: float = 0.0) -> None:
        self.scorer, self.threshold = scorer, threshold

    def verify(self, answer: str, passages: list[str], tool_observations: list[str]) -> list[dict[str, Any]]:
        out = []
        for claim in split_claims(answer):
            scores = self.scorer(claim, passages) if (self.scorer and passages) else []
            d = decide_support(claim, scores, tool_observations, self.threshold)
            out.append({"claim": claim, **d})
        return out


class Agent:
    def __init__(self, generator: Any, tools: dict[str, Callable[..., str]], tracer: Tracer | None = None, max_steps: int = 4, verifier: Verifier | None = None, verify_mode: str = "flag", native_tools: bool = False, evidence_source: Any = None, max_new_tokens: int = 96) -> None:
        self.generator, self.tools, self.tracer = generator, tools, tracer or Tracer()
        self.max_steps, self.verifier, self.verify_mode, self.native_tools = max_steps, verifier, verify_mode, native_tools
        self.evidence_source = evidence_source  # SearchTool instance exposing `.evidence`
        self.max_new_tokens = max_new_tokens
        self.specs = [s for s in TOOL_SPECS if s["name"] in tools]

    def _call_tool(self, name: str, args: dict[str, Any]) -> str:
        fn = self.tools.get(name)
        if fn is None:
            return f"error: unknown tool {name}; available: {', '.join(self.tools)}"
        try:
            return str(fn(**args))
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"

    def run(self, task: str) -> dict[str, Any]:
        tr = self.tracer
        if self.evidence_source is not None:
            self.evidence_source.evidence = []
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT.format(tools=render_tool_help(self.specs))}, {"role": "user", "content": task}]
        steps: list[dict[str, Any]] = []
        observations: list[str] = []
        ptoks = otoks = 0
        answer = None
        with tr.span("agent.task", "other", task=task):
            for step in range(self.max_steps + 1):
                force_final = step == self.max_steps
                if force_final:
                    messages.append({"role": "user", "content": 'You must answer now. Respond with {"thought": "...", "action": "final", "answer": "..."}.'})
                with tr.span("model", "model", step=step) as s:
                    g = self.generator.generate(messages, max_new_tokens=self.max_new_tokens, tools=openai_tool_defs(self.specs) if self.native_tools else None)
                    ptoks += g["prompt_tokens"]
                    otoks += g["output_tokens"]
                    s.set(prompt_tokens=g["prompt_tokens"], output_tokens=g["output_tokens"])
                if g.get("tool_calls"):
                    tc = g["tool_calls"][0]
                    try:
                        args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else dict(tc["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    act = {"action": tc["name"], "args": args, "thought": g["text"]}
                else:
                    act = parse_action(g["text"]) or {"action": "final", "answer": g["text"].strip(), "thought": None, "parse_failed": True}
                messages.append({"role": "assistant", "content": g["text"] or json.dumps(act)})
                if act["action"] == "final" or force_final:
                    if act["action"] != "final" and observations:
                        answer = observations[-1]  # forced final but the model asked for another tool: use the last observation
                    else:
                        answer = str(act.get("answer", g["text"])).strip()
                    steps.append({"step": step, "action": "final", "answer": answer, "parse_failed": act.get("parse_failed", False), "salvaged": act.get("salvaged", False), "forced": force_final})
                    break
                with tr.span(f"tool:{act['action']}", "tool", **{k: str(v)[:80] for k, v in act["args"].items()}) as s:
                    t0 = time.perf_counter()
                    obs = self._call_tool(act["action"], act["args"])
                    s.set(wait_s=time.perf_counter() - t0, chars=len(obs))
                observations.append(obs)
                steps.append({"step": step, "action": act["action"], "args": act["args"], "observation": obs[:300]})
                messages.append({"role": "user", "content": f"Observation: {obs}\nContinue. Respond with one JSON object."})
            verification = None
            final_answer = answer or ""
            if self.verifier is not None:
                passages = [e["passage"] for e in (self.evidence_source.evidence if self.evidence_source is not None else [])]
                with tr.span("verify", "verify") as s:
                    verification = self.verifier.verify(final_answer, passages, observations)
                    unsupported = [v for v in verification if not v["supported"]]
                    s.set(n_claims=len(verification), n_unsupported=len(unsupported))
                if unsupported and self.verify_mode == "remove":
                    kept = [v["claim"] for v in verification if v["supported"]]
                    final_answer = " ".join(kept) if kept else "unknown"
                elif unsupported and self.verify_mode == "flag":
                    final_answer = f"{final_answer} [unverified: {len(unsupported)}/{len(verification)} claims lack evidence]"
        bd = tr.breakdown()
        return {
            "task": task,
            "raw_answer": answer,
            "answer": final_answer,
            "steps": steps,
            "n_model_calls": sum(1 for s in tr.spans if s.kind == "model"),
            "n_tool_calls": sum(1 for s in tr.spans if s.kind == "tool"),
            "prompt_tokens": ptoks,
            "output_tokens": otoks,
            "verification": verification,
            "latency_s": {"wall": bd["_wall_s"], "model": bd["model"]["total_s"], "tool": bd["tool"]["total_s"], "verify": bd["verify"]["total_s"]},
        }
