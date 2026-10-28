"""L03: SLO analysis and sweep aggregation on synthetic run directories (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_systems_zoo import SLO, RequestRecord, Run
from llm_systems_zoo.metrics import full_report
from serving.run_sweep import main as sweep_main
from serving.slo_analysis import analyze, markdown_table, plot_saturation, record_from_dict
from serving.slo_analysis import main as analysis_main


def _fake_run(root: Path, load: float, ttft: float, n: int, label: str = "A") -> Path:
    recs = []
    for i in range(n):
        t0 = i / load
        recs.append(RequestRecord(f"r{i}", t0, 50, 11, t0 + ttft, t0 + ttft + 0.5, None, {"request_class": "a"}))
    with Run("LXX", f"r{load}", {"load": {"mode": "open", "rate": load}, "label": label}, root=root, collect_env=False) as run:
        run.log_records([r.to_dict() for r in recs])
        rep = full_report(recs, SLO(ttft_s=0.2))
        rep["offered_load"] = load
        rep["mode"] = "open"
        rep["meta"] = {"max_in_flight_observed": 3, "rejected_admission": 0}
        run.save_summary(rep)
    return run.dir


def test_record_round_trip():
    r = RequestRecord("x", 1.0, 10, 5, 1.2, 2.0, None, {"request_class": "a"})
    back = record_from_dict(r.to_dict())
    assert back.ttft == r.ttft and back.tpot == r.tpot and back.extra["request_class"] == "a"


def test_analyze_and_table(tmp_path: Path):
    dirs = [_fake_run(tmp_path, 2.0, 0.05, 150), _fake_run(tmp_path, 8.0, 0.5, 40)]
    rows = analyze(dirs, SLO(ttft_s=0.2))
    assert [r["offered_load"] for r in rows] == [2.0, 8.0]
    assert rows[0]["slo_attainment"] == 1.0 and rows[1]["slo_attainment"] == 0.0
    assert rows[0]["p99_undersampled"] is False and rows[1]["p99_undersampled"] is True
    table = markdown_table(rows, SLO(ttft_s=0.2))
    assert "under-sampled" in table and "| A | open | 2 |" in table
    out = plot_saturation(rows, tmp_path / "fig.png", "t", SLO(ttft_s=0.2))
    assert out.exists() and out.stat().st_size > 1000


def test_analysis_cli_writes_outputs(tmp_path: Path, capsys):
    d1 = _fake_run(tmp_path / "runs", 1.0, 0.05, 20)
    d2 = _fake_run(tmp_path / "runs", 4.0, 0.05, 20)
    rows = analysis_main([str(d1), str(d2), "--slo", "ttft=0.1", "--out", str(tmp_path / "out")])
    assert len(rows) == 2
    assert (tmp_path / "out" / "saturation.md").exists() and (tmp_path / "out" / "saturation.json").exists()
    assert len(json.loads((tmp_path / "out" / "saturation.json").read_text())) == 2


def test_fake_rate_sweep_end_to_end(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "benchmarks" / "workloads").mkdir(parents=True)
    from serving.workloads import builtin_specs

    builtin_specs()["short_qa"].save(tmp_path / "benchmarks" / "workloads" / "short_qa.yaml")
    sweep_main(["rate-sweep", "--fake", "--smoke", "--rates", "20,60", "--fake-capacity", "4", "--fake-ttft", "0.01", "--fake-tpot", "0.001",
                "--warmup", "1", "--report-dir", str(tmp_path / "report")])
    sweep = json.loads((tmp_path / "report" / "sweep.json").read_text())
    assert len(sweep["rows"]) == 2
    assert sweep["rows"][1]["ttft_p99"] > sweep["rows"][0]["ttft_p99"]  # heavier load -> longer queue
    assert (tmp_path / "report" / "saturation.png").exists()


def test_fake_overload_with_admission(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "benchmarks" / "workloads").mkdir(parents=True)
    from serving.workloads import builtin_specs

    builtin_specs()["short_qa"].save(tmp_path / "benchmarks" / "workloads" / "short_qa.yaml")
    sweep_main(["overload", "--fake", "--smoke", "--rate", "30", "--factor", "2", "--max-in-flight", "6", "--fake-capacity", "4",
                "--fake-ttft", "0.01", "--fake-tpot", "0.001", "--warmup", "0", "--slo", "ttft=0.1"])
    sweeps = list((tmp_path / "runs" / "L03_serving_slo" / "sweeps").iterdir())
    rows = json.loads((sweeps[0] / "sweep.json").read_text())["rows"]
    by = {r["label"]: r for r in rows}
    assert by["max_in_flight_6"]["rejected"] > 0 and by["no_admission"]["rejected"] == 0
    assert by["max_in_flight_6"]["ttft_p99"] < by["no_admission"]["ttft_p99"]
