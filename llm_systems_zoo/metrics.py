"""Latency, throughput and SLO-goodput metrics shared by the serving/inference labs.

Definitions (fixed here so every lab reports the same thing):

* TTFT  time to first token      = t_first_token - t_submit
* TPOT  time per output token    = (t_end - t_first_token) / (output_tokens - 1)
* E2E   end-to-end latency       = t_end - t_submit
* Throughput is measured over the observed window [min(t_submit), max(t_end)], and
  input and output token throughput are reported separately.
* SLO goodput = completed requests meeting every SLO condition / observed window (req/s).
  SLO attainment = that count / all *submitted* requests, so failures and time-outs
  count against the configuration instead of being silently dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass
class RequestRecord:
    request_id: str
    t_submit: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    t_first_token: float | None = None
    t_end: float | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None and self.t_end is not None

    @property
    def ttft(self) -> float | None:
        return None if self.t_first_token is None else self.t_first_token - self.t_submit

    @property
    def e2e(self) -> float | None:
        return None if self.t_end is None else self.t_end - self.t_submit

    @property
    def tpot(self) -> float | None:
        if self.t_first_token is None or self.t_end is None or self.output_tokens < 2:
            return None
        return (self.t_end - self.t_first_token) / (self.output_tokens - 1)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RequestRecord:
        """Inverse of `to_dict` (derived fields ttft/tpot/e2e are ignored; unknown keys go to `extra`)."""
        known = {"request_id", "t_submit", "t_first_token", "t_end", "prompt_tokens", "output_tokens", "error"}
        derived = {"ttft", "tpot", "e2e"}
        return cls(
            request_id=str(d["request_id"]),
            t_submit=float(d["t_submit"]),
            prompt_tokens=int(d.get("prompt_tokens") or 0),
            output_tokens=int(d.get("output_tokens") or 0),
            t_first_token=None if d.get("t_first_token") is None else float(d["t_first_token"]),
            t_end=None if d.get("t_end") is None else float(d["t_end"]),
            error=d.get("error"),
            extra={k: v for k, v in d.items() if k not in known and k not in derived},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "t_submit": self.t_submit,
            "t_first_token": self.t_first_token,
            "t_end": self.t_end,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
            "ttft": self.ttft,
            "tpot": self.tpot,
            "e2e": self.e2e,
            **self.extra,
        }


def percentile(values: Sequence[float], p: float) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=float), p))


def summarize(values: Sequence[float], percentiles: Sequence[float] = (50, 90, 95, 99)) -> dict[str, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    out = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    for p in percentiles:
        out[f"p{int(p) if float(p).is_integer() else p}"] = float(np.percentile(arr, p))
    return out


def latency_summary(records: Iterable[RequestRecord]) -> dict[str, Any]:
    recs = list(records)
    ok = [r for r in recs if r.success]
    return {
        "n_submitted": len(recs),
        "n_success": len(ok),
        "n_failed": len(recs) - len(ok),
        "error_rate": (len(recs) - len(ok)) / len(recs) if recs else math.nan,
        "ttft_s": summarize([r.ttft for r in ok if r.ttft is not None]),
        "tpot_s": summarize([r.tpot for r in ok if r.tpot is not None]),
        "e2e_s": summarize([r.e2e for r in ok if r.e2e is not None]),
        "prompt_tokens": summarize([r.prompt_tokens for r in ok], percentiles=(50, 99)),
        "output_tokens": summarize([r.output_tokens for r in ok], percentiles=(50, 99)),
    }


def observed_window(records: Iterable[RequestRecord]) -> float:
    recs = list(records)
    if not recs:
        return 0.0
    start = min(r.t_submit for r in recs)
    ends = [r.t_end for r in recs if r.t_end is not None]
    end = max(ends) if ends else start
    return max(end - start, 1e-9)


def throughput(records: Iterable[RequestRecord], window_s: float | None = None) -> dict[str, float]:
    recs = list(records)
    ok = [r for r in recs if r.success]
    window = window_s or observed_window(recs)
    return {
        "window_s": window,
        "requests_per_s": len(ok) / window,
        "output_tokens_per_s": sum(r.output_tokens for r in ok) / window,
        "input_tokens_per_s": sum(r.prompt_tokens for r in ok) / window,
    }


@dataclass
class SLO:
    """A request meets the SLO when every non-None bound holds."""

    ttft_s: float | None = None
    tpot_s: float | None = None
    e2e_s: float | None = None

    def met(self, r: RequestRecord) -> bool:
        if not r.success:
            return False
        if self.ttft_s is not None and (r.ttft is None or r.ttft > self.ttft_s):
            return False
        if self.tpot_s is not None and r.tpot is not None and r.tpot > self.tpot_s:
            return False
        if self.e2e_s is not None and (r.e2e is None or r.e2e > self.e2e_s):
            return False
        return True

    def to_dict(self) -> dict[str, float | None]:
        return {"ttft_s": self.ttft_s, "tpot_s": self.tpot_s, "e2e_s": self.e2e_s}


def goodput(records: Iterable[RequestRecord], slo: SLO, window_s: float | None = None) -> dict[str, float]:
    recs = list(records)
    window = window_s or observed_window(recs)
    met = [r for r in recs if slo.met(r)]
    return {
        "slo": slo.to_dict(),
        "n_met": len(met),
        "goodput_rps": len(met) / window,
        "goodput_output_tokens_per_s": sum(r.output_tokens for r in met) / window,
        "slo_attainment": len(met) / len(recs) if recs else math.nan,
    }


def full_report(records: Iterable[RequestRecord], slo: SLO | None = None, window_s: float | None = None) -> dict[str, Any]:
    recs = list(records)
    rep = {"latency": latency_summary(recs), "throughput": throughput(recs, window_s)}
    if slo is not None:
        rep["goodput"] = goodput(recs, slo, window_s)
    return rep


def saturation_curve(points: Iterable[tuple[float, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten (offered_load, full_report) pairs into rows for a table / plot."""
    rows = []
    for load, rep in points:
        row = {
            "offered_load": load,
            "requests_per_s": rep["throughput"]["requests_per_s"],
            "output_tokens_per_s": rep["throughput"]["output_tokens_per_s"],
            "ttft_p50": rep["latency"]["ttft_s"].get("p50"),
            "ttft_p99": rep["latency"]["ttft_s"].get("p99"),
            "tpot_p50": rep["latency"]["tpot_s"].get("p50"),
            "tpot_p99": rep["latency"]["tpot_s"].get("p99"),
            "e2e_p99": rep["latency"]["e2e_s"].get("p99"),
            "error_rate": rep["latency"]["error_rate"],
        }
        if "goodput" in rep:
            row["goodput_rps"] = rep["goodput"]["goodput_rps"]
            row["slo_attainment"] = rep["goodput"]["slo_attainment"]
        rows.append(row)
    return rows
