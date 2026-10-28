"""L03: the load generator's bookkeeping against the in-process fake OpenAI server (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_systems_zoo.io import read_jsonl
from serving.fake_server import FakeEngine, FakeServerThread
from serving.load_generator import LoadConfig, format_summary, parse_slo, run_load
from serving.workloads import ArrivalSpec, LengthDist, RequestClass, WorkloadSpec


@pytest.fixture(scope="module")
def server():
    with FakeServerThread(FakeEngine(max_concurrency=4, ttft_s=0.02, tpot_s=0.002)) as srv:
        yield srv


def _spec(n: int = 12, out: int = 10) -> WorkloadSpec:
    return WorkloadSpec(
        name="t", seed=1, num_requests=n,
        classes=[RequestClass("a", prompt_len=LengthDist("fixed", 20), output_len=LengthDist("fixed", out))],
        arrival=ArrivalSpec(mode="closed", concurrency=2),
    )


def test_closed_loop_records_and_report(server, tmp_path: Path):
    cfg = LoadConfig(endpoint=server.url, model="fake", mode="closed", concurrency=3, max_requests=12, warmup=2, ignore_eos=False)
    run = run_load(cfg, _spec(), slo=parse_slo("ttft=0.5,tpot=0.05"), run_root=tmp_path, collect_env=False, quiet=True)
    recs = list(read_jsonl(run.dir / "records.jsonl"))
    assert len(recs) == 12  # warm-up excluded
    assert all(r["error"] is None for r in recs)
    assert all(r["output_tokens"] == 10 for r in recs)  # from the usage field
    assert all(r["prompt_tokens"] == 20 for r in recs)
    assert all(r["ttft"] >= 0.02 for r in recs)
    assert all(0.0015 < r["tpot"] < 0.02 for r in recs)
    summary = json.loads((run.dir / "summary.json").read_text())
    assert summary["latency"]["n_success"] == 12
    assert summary["goodput"]["slo_attainment"] == 1.0
    assert summary["meta"]["max_in_flight_observed"] <= 3
    assert summary["p99_undersampled"] is True
    assert "goodput" in format_summary(summary)
    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert manifest["execution"]["measured_requests"] == 12 and manifest["execution"]["warmup_requests"] == 2


def test_chat_api_path(server, tmp_path: Path):
    cfg = LoadConfig(endpoint=server.url, model="fake", api="chat", mode="closed", concurrency=2, max_requests=4, warmup=0, ignore_eos=False, system_prompt="sys")
    run = run_load(cfg, _spec(), run_root=tmp_path, collect_env=False, quiet=True)
    recs = list(read_jsonl(run.dir / "records.jsonl"))
    assert len(recs) == 4 and all(r["error"] is None and r["output_tokens"] == 10 for r in recs)


def test_open_loop_admission_control_rejects_excess(server, tmp_path: Path):
    # 200 req/s against a 4-slot server: with max_in_flight=5 most arrivals must be rejected.
    cfg = LoadConfig(endpoint=server.url, model="fake", mode="open", rate=200, duration_s=0.5, warmup=0, max_in_flight=5, ignore_eos=False)
    run = run_load(cfg, _spec(out=20), slo=parse_slo("ttft=0.1"), run_root=tmp_path, collect_env=False, quiet=True)
    summary = json.loads((run.dir / "summary.json").read_text())
    recs = list(read_jsonl(run.dir / "records.jsonl"))
    rejected = [r for r in recs if r["error"] == "rejected_admission"]
    assert len(recs) == 100 and len(rejected) > 50
    assert summary["meta"]["max_in_flight_observed"] <= 5
    assert summary["latency"]["n_submitted"] == 100 and summary["latency"]["n_failed"] == len(rejected)
    # attainment is over submitted requests, so rejections pull it down
    assert summary["goodput"]["slo_attainment"] <= summary["latency"]["n_success"] / 100


def test_open_loop_without_admission_queues_instead(server, tmp_path: Path):
    cfg = LoadConfig(endpoint=server.url, model="fake", mode="open", rate=100, duration_s=0.3, warmup=0, ignore_eos=False)
    run = run_load(cfg, _spec(out=20), run_root=tmp_path, collect_env=False, quiet=True)
    summary = json.loads((run.dir / "summary.json").read_text())
    assert summary["latency"]["n_failed"] == 0
    assert summary["meta"]["max_in_flight_observed"] > 4  # queue builds up beyond server capacity
    assert summary["latency"]["ttft_s"]["p99"] > summary["latency"]["ttft_s"]["min"] * 2


def test_timeout_is_recorded_as_error(tmp_path: Path):
    with FakeServerThread(FakeEngine(max_concurrency=1, ttft_s=0.5, tpot_s=0.0)) as slow:
        cfg = LoadConfig(endpoint=slow.url, model="fake", mode="closed", concurrency=1, max_requests=2, warmup=0, timeout_s=0.1, ignore_eos=False)
        run = run_load(cfg, _spec(), run_root=tmp_path, collect_env=False, quiet=True)
    recs = list(read_jsonl(run.dir / "records.jsonl"))
    assert all(r["error"] == "timeout" for r in recs)


def test_http_error_is_recorded(tmp_path: Path):
    with FakeServerThread(FakeEngine(max_concurrency=1, queue_limit=0)) as full:
        cfg = LoadConfig(endpoint=full.url, model="fake", mode="open", rate=50, duration_s=0.1, warmup=0, ignore_eos=False)
        run = run_load(cfg, _spec(), run_root=tmp_path, collect_env=False, quiet=True)
    recs = list(read_jsonl(run.dir / "records.jsonl"))
    assert any(r["error"] and r["error"].startswith("http_503") for r in recs)


def test_parse_slo():
    s = parse_slo("ttft=0.5,tpot=0.05,e2e=10")
    assert s.ttft_s == 0.5 and s.tpot_s == 0.05 and s.e2e_s == 10
    assert parse_slo("") is None and parse_slo("ttft=1").tpot_s is None
