"""A tiny span tracer for the application lab.

Spans are nested context managers with a `kind` from {retrieval, rerank, model, tool, verify, other}
so the latency of an agent/RAG request can be decomposed into *model compute* vs *tool wait* vs
*retrieval*. The tracer exports plain JSON and the Chrome trace-event format (open it at
https://ui.perfetto.dev). In production the same role is played by OpenTelemetry spans; this file
exists so the lab has no dependency and the semantics are explicit.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

KINDS = ("retrieval", "rerank", "model", "tool", "verify", "other")


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    kind: str
    t_start: float
    t_end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return (self.t_end if self.t_end is not None else time.perf_counter()) - self.t_start

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)


class Tracer:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, kind: str = "other", **attrs: Any) -> Iterator[Span]:
        if kind not in KINDS:
            raise ValueError(f"unknown span kind {kind!r}; expected one of {KINDS}")
        s = Span(uuid.uuid4().hex[:8], self._stack[-1].span_id if self._stack else None, name, kind, time.perf_counter(), attributes=dict(attrs))
        self.spans.append(s)
        self._stack.append(s)
        try:
            yield s
        finally:
            self._stack.pop()
            s.t_end = time.perf_counter()

    # ---- analysis --------------------------------------------------------------------
    def wall_s(self) -> float:
        if not self.spans:
            return 0.0
        return max(s.t_end or s.t_start for s in self.spans) - min(s.t_start for s in self.spans)

    def _ancestor_has_kind(self, s: Span, kind: str) -> bool:
        by_id = {x.span_id: x for x in self.spans}
        p = s.parent_id
        while p is not None:
            parent = by_id[p]
            if parent.kind == kind:
                return True
            p = parent.parent_id
        return False

    def breakdown(self) -> dict[str, Any]:
        """Seconds per kind. A span nested inside a span of the *same* kind is not double counted."""
        totals = {k: 0.0 for k in KINDS}
        counts = {k: 0 for k in KINDS}
        for s in self.spans:
            if s.t_end is None or self._ancestor_has_kind(s, s.kind):
                continue
            totals[s.kind] += s.duration_s
            counts[s.kind] += 1
        wall = self.wall_s()
        out: dict[str, Any] = {k: {"total_s": totals[k], "count": counts[k], "share": totals[k] / wall if wall else 0.0} for k in KINDS}
        out["_wall_s"] = wall
        return out

    # ---- export ----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "spans": [s.__dict__ | {"duration_s": s.duration_s} for s in self.spans], "breakdown": self.breakdown()}

    def to_chrome_trace(self) -> dict[str, Any]:
        t0 = min((s.t_start for s in self.spans), default=0.0)
        events = []
        for s in self.spans:
            events.append({"name": s.name, "cat": s.kind, "ph": "X", "ts": (s.t_start - t0) * 1e6, "dur": s.duration_s * 1e6, "pid": 1, "tid": 1, "args": {"span_id": s.span_id, "parent_id": s.parent_id, **s.attributes}})
        return {"traceEvents": events, "displayTimeUnit": "ms"}

    def save(self, path: str | Path, chrome: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        if chrome:
            path.with_suffix(".chrome.json").write_text(json.dumps(self.to_chrome_trace(), default=str))
        return path

    def reset(self) -> None:
        self.spans.clear()
        self._stack.clear()
        self.trace_id = uuid.uuid4().hex[:12]
