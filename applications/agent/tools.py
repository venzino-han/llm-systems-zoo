"""Tools for the minimal agent: a safe calculator, a date lookup, and passage search over the
RAG index. Each tool has a JSON-schema spec usable both for prompt rendering (JSON-action
protocol) and for native OpenAI-style tool calling."""

from __future__ import annotations

import ast
import datetime as dt
import operator
import re
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 64:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression (+ - * / // % ** and parentheses only)."""
    expr = re.sub(r"^(what is|calculate|compute)\s+", "", str(expression).strip(), flags=re.I)
    expr = expr.rstrip(" =?.").replace("×", "*").replace("÷", "/").replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
        val = _eval(tree)
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return str(val)


def lookup_date(fixed: str | None = None) -> str:
    """Today's date (ISO). `fixed` makes evaluation deterministic."""
    return fixed or dt.date.today().isoformat()


class SearchTool:
    """Search the RAG index; returns the top passages as numbered text and keeps them as evidence."""

    def __init__(self, retriever, reranker=None, k: int = 10, top_n: int = 3) -> None:
        self.retriever, self.reranker, self.k, self.top_n = retriever, reranker, k, top_n
        self.evidence: list[dict[str, Any]] = []

    def __call__(self, query: str) -> str:
        cands = self.retriever.retrieve(str(query), self.k)
        hits = self.reranker.rerank(query, cands, self.top_n) if self.reranker else cands[: self.top_n]
        self.evidence.extend(hits)
        return "\n".join(f"[{i + 1}] {h['passage'][:600]}" for i, h in enumerate(hits))


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "search_passages",
        "description": "Search an encyclopedia corpus. Use it for any factual question about people, places, dates or events.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "search query"}}, "required": ["query"]},
    },
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly, e.g. '123*45 + 6'.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    },
    {
        "name": "lookup_date",
        "description": "Return today's date in ISO format.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def openai_tool_defs(specs: list[dict[str, Any]] = TOOL_SPECS) -> list[dict[str, Any]]:
    return [{"type": "function", "function": s} for s in specs]


def render_tool_help(specs: list[dict[str, Any]] = TOOL_SPECS) -> str:
    lines = []
    for s in specs:
        params = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in s["parameters"].get("properties", {}).items()) or "no arguments"
        lines.append(f"- {s['name']}({params}): {s['description']}")
    return "\n".join(lines)
