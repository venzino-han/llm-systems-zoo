"""L04: topology interpretation, proxy routing, minimal TP equivalence, dry-run planning (CPU)."""

from __future__ import annotations

import re

import pytest

from inference.multi_gpu.bench_tp_vs_replica import layout_configs, proxy_command, render_plan
from inference.multi_gpu.bench_tp_vs_replica import main as bench_main
from inference.multi_gpu.proxy import RoundRobin
from inference.multi_gpu.topology import classify, summarize_topology
from serving.server import ServerConfig

CANNED_TOPO = {"GPU0-GPU1": "PHB", "GPU1-GPU0": "PHB"}
CANNED_GPUS = [{"index": 0, "name": "NVIDIA RTX A6000"}, {"index": 1, "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"}]


def test_classify_links():
    assert classify("NV12")[0] == "NVLink"
    assert classify("PHB")[0].startswith("PCIe via host")
    assert classify("SYS")[1] < classify("PIX")[1]
    assert classify("??") == ("unknown", 0.0)


def test_summarize_topology_pcie_heterogeneous():
    s = summarize_topology(CANNED_TOPO, CANNED_GPUS)
    assert s["pairs"] == [{"pair": "GPU0-GPU1", "link": "PHB", "class": "PCIe via host bridge (CPU)", "nominal_gbps_bidir": 64.0}]
    assert s["has_nvlink"] is False and s["heterogeneous"] is True
    assert any("no NVLink" in a for a in s["advice"]) and any("heterogeneous" in a for a in s["advice"])


def test_summarize_topology_nvlink_single_gpu():
    s = summarize_topology({"GPU0-GPU1": "NV4", "GPU1-GPU0": "NV4"}, [{"index": 0, "name": "A"}, {"index": 1, "name": "A"}])
    assert s["has_nvlink"] and not s["heterogeneous"]
    assert summarize_topology({}, [{"index": 0, "name": "A"}])["advice"][0].startswith("single GPU")


def test_round_robin_cycles_and_counts():
    rr = RoundRobin(["http://a/", "http://b"])
    assert [rr.next() for _ in range(5)] == ["http://a", "http://b", "http://a", "http://b", "http://a"]
    assert rr.counts == [3, 2]
    with pytest.raises(ValueError):
        RoundRobin([])


def test_layout_configs_and_plan():
    base = ServerConfig(model="m", gpu_memory_utilization=0.5, max_model_len=1024)
    single = layout_configs("single", [0, 1], base, 8030)
    tp = layout_configs("tp", [0, 1], base, 8030)
    rep = layout_configs("replica", [0, 1], base, 8030)
    assert len(single) == 1 and single[0].env["CUDA_VISIBLE_DEVICES"] == "0" and single[0].tensor_parallel_size == 1
    assert tp[0].tensor_parallel_size == 2 and tp[0].env["CUDA_VISIBLE_DEVICES"] == "0,1" and "--tensor-parallel-size 2" in " ".join(tp[0].command())
    assert [c.port for c in rep] == [8031, 8032] and [c.env["CUDA_VISIBLE_DEVICES"] for c in rep] == ["0", "1"]
    plan = render_plan(["single", "tp", "replica"], [0, 1], base, 8030)
    assert [p["n_gpus"] for p in plan] == [1, 2, 2]
    assert plan[2]["endpoint"] == "http://127.0.0.1:8030" and len(plan[2]["commands"]) == 3
    assert "inference.multi_gpu.proxy" in plan[2]["commands"][-1]
    assert proxy_command(8030, ["http://x"])[-1] == "http://x"
    with pytest.raises(ValueError):
        layout_configs("pp", [0, 1], base, 8030)


def test_dry_run_prints_commands(capsys):
    bench_main(["--dry-run", "--gpus", "0,1", "--model", "Qwen/Qwen2.5-0.5B-Instruct", "--server-config", ""])
    out = capsys.readouterr().out
    assert "layout tp" in out and "--tensor-parallel-size 2" in out
    assert len(re.findall(r"api_server", out)) == 4  # single + tp + 2 replicas


def test_minimal_tp_matches_dense_on_cpu():
    from inference.multi_gpu.minimal_tp import run_spawned

    res = run_spawned(2, "gloo", hidden=128, ffn=512, tokens=32, iters=2)
    assert res["world"] == 2 and res["max_abs_err"] < 1e-4
    assert len(res["rows"]) == 4 and all(r["comm_ms_per_block"] >= 0 for r in res["rows"])


def test_comm_bench_gloo_smoke():
    from inference.multi_gpu.comm_bench import tp_decode_floor

    rows = [{"op": "all_reduce", "bytes": 4096, "mean_us": 100.0}, {"op": "all_reduce", "bytes": 1 << 20, "mean_us": 500.0}]
    f = tp_decode_floor(rows, layers=28, hidden=3584, batch=1, dtype_bytes=2)
    expected_us = 100.0 + (7168 - 4096) / ((1 << 20) - 4096) * 400.0  # linear between the two sizes
    assert f["message_bytes"] == 7168 and f["all_reduce_us"] == pytest.approx(expected_us)
    assert f["comm_floor_per_token_ms"] == pytest.approx(56 * expected_us / 1000)
    assert tp_decode_floor(rows, batch=1000)["all_reduce_us"] == 500.0  # clamped at the largest size
