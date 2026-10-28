"""Tests for the shared harness: manifests, metrics, timers. CPU only, no network."""

from __future__ import annotations

import json
import math
from pathlib import Path

from llm_systems_zoo import (
    SLO,
    PhaseTimer,
    RequestRecord,
    Run,
    goodput,
    latency_summary,
    throughput,
)
from llm_systems_zoo.io import read_jsonl, sha256_of
from llm_systems_zoo.manifest import config_hash
from llm_systems_zoo.metrics import full_report, saturation_curve, summarize


def _records(n: int = 10, fail_every: int = 0) -> list[RequestRecord]:
    recs = []
    for i in range(n):
        t0 = float(i)
        err = "timeout" if fail_every and i % fail_every == 0 else None
        recs.append(
            RequestRecord(
                request_id=f"r{i}",
                t_submit=t0,
                prompt_tokens=100,
                output_tokens=11,
                t_first_token=None if err else t0 + 0.1 * (i + 1),
                t_end=None if err else t0 + 0.1 * (i + 1) + 1.0,  # 10 decode steps of 0.1s
                error=err,
            )
        )
    return recs


def test_request_record_derived_latencies():
    r = _records(1)[0]
    assert math.isclose(r.ttft, 0.1)
    assert math.isclose(r.tpot, 0.1)  # 1.0s over 10 inter-token gaps
    assert math.isclose(r.e2e, 1.1)
    assert r.success


def test_failed_request_counts_against_error_rate_and_goodput():
    recs = _records(10, fail_every=5)  # requests 0 and 5 fail
    lat = latency_summary(recs)
    assert lat["n_submitted"] == 10 and lat["n_failed"] == 2
    assert math.isclose(lat["error_rate"], 0.2)
    g = goodput(recs, SLO(ttft_s=10, tpot_s=10), window_s=10.0)
    assert g["n_met"] == 8
    assert math.isclose(g["slo_attainment"], 0.8)
    assert math.isclose(g["goodput_rps"], 0.8)


def test_slo_bounds_filter_requests():
    recs = _records(10)
    # TTFT grows with i: 0.1 .. 1.0; a 0.55s bound admits i=0..4
    g = goodput(recs, SLO(ttft_s=0.55), window_s=10.0)
    assert g["n_met"] == 5


def test_throughput_uses_observed_window_and_separates_token_directions():
    recs = _records(10)
    th = throughput(recs)
    assert math.isclose(th["window_s"], 9.0 + 1.0 + 1.0)  # last submit 9.0, +ttft 1.0, +1.0 decode
    assert math.isclose(th["output_tokens_per_s"] * th["window_s"], 110)
    assert math.isclose(th["input_tokens_per_s"] * th["window_s"], 1000)


def test_summarize_percentiles():
    s = summarize(list(range(1, 101)))
    assert s["n"] == 100 and math.isclose(s["p50"], 50.5) and math.isclose(s["p99"], 99.01)
    assert summarize([]) == {"n": 0}


def test_saturation_curve_rows():
    rows = saturation_curve([(1.0, full_report(_records(5), SLO(ttft_s=1))), (2.0, full_report(_records(5)))])
    assert rows[0]["offered_load"] == 1.0 and "goodput_rps" in rows[0]
    assert "goodput_rps" not in rows[1]


def test_run_creates_manifest_and_logs(tmp_path: Path):
    cfg = {"model": "tiny", "batch": 2}
    with Run("LXX_test", "smoke", cfg, root=tmp_path, collect_env=False, model={"id": "x"}) as run:
        run.log_metric(step=1, loss=1.0)
        run.log_metric(step=2, loss=0.5)
        run.log_records(_records(2))
        run.save_summary({"ok": True})
        run.note("hello")
        run.update(execution={"warmup": 1})
    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["config_hash"] == config_hash(cfg) == sha256_of(cfg)
    assert manifest["model"] == {"id": "x"} and manifest["execution"] == {"warmup": 1}
    assert manifest["notes"] == ["hello"] and manifest["duration_s"] is not None
    assert [m["step"] for m in read_jsonl(run.dir / "metrics.jsonl")] == [1, 2]
    assert len(list(read_jsonl(run.dir / "records.jsonl"))) == 2
    assert json.loads((run.dir / "summary.json").read_text()) == {"ok": True}
    assert run.dir.name.endswith(config_hash(cfg))


def test_run_marks_failure_on_exception(tmp_path: Path):
    try:
        with Run("LXX_test", "boom", {}, root=tmp_path, collect_env=False) as run:
            raise RuntimeError("x")
    except RuntimeError:
        pass
    assert json.loads((run.dir / "manifest.json").read_text())["status"] == "failed"


def test_phase_timer_accumulates():
    t = PhaseTimer()
    with t.phase("a"):
        pass
    with t.phase("a"):
        pass
    t.add("b", 0.5)
    s = t.summary()
    assert s["a"]["count"] == 2 and math.isclose(s["b"]["total_s"], 0.5)
    assert "_wall_s" in s and "_unaccounted_s" in s


def test_request_record_round_trip():
    r = _records(1)[0]
    r.extra["cls"] = "short"
    back = RequestRecord.from_dict(r.to_dict())
    assert back.request_id == r.request_id and back.t_end == r.t_end and back.extra == {"cls": "short"}
    assert math.isclose(back.tpot, r.tpot)


def test_copy_results(tmp_path: Path):
    from llm_systems_zoo import copy_results

    with Run("LXX_test", "copy", {"a": 1}, root=tmp_path / "runs", collect_env=False) as run:
        run.save_summary({"x": 1})
        (run.dir / "fig.png").write_bytes(b"png")
        (run.dir / "records.jsonl").write_text("{}\n")
    dest = copy_results(run.dir, tmp_path / "reports" / "LXX" / "results")
    names = sorted(p.name for p in dest.iterdir())
    assert names == ["config.json", "fig.png", "manifest.json", "summary.json"]


def test_log_metric_keeps_nested_values(tmp_path: Path):
    with Run("LXX_test", "nested", {}, root=tmp_path, collect_env=False) as run:
        run.log_metric(step=1, phases={"a": 1.0, "b": [1, 2]})
    rows = list(read_jsonl(run.dir / "metrics.jsonl"))
    assert rows[0]["phases"] == {"a": 1.0, "b": [1, 2]}
