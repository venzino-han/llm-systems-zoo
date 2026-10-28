from __future__ import annotations

import json
import time

import pytest

from applications.trace import Tracer


def test_nesting_and_breakdown_without_double_counting():
    tr = Tracer()
    with tr.span("request", "other"):
        with tr.span("retrieve", "retrieval"):
            time.sleep(0.01)
        with tr.span("model", "model"):
            time.sleep(0.01)
            with tr.span("model.inner", "model"):  # same kind nested: not double counted
                time.sleep(0.01)
    assert tr.spans[1].parent_id == tr.spans[0].span_id and tr.spans[3].parent_id == tr.spans[2].span_id
    bd = tr.breakdown()
    assert bd["model"]["count"] == 1 and bd["retrieval"]["count"] == 1
    assert bd["model"]["total_s"] >= 0.02 and bd["model"]["total_s"] < bd["_wall_s"] + 1e-6
    assert bd["retrieval"]["total_s"] >= 0.01


def test_chrome_export_and_save(tmp_path):
    tr = Tracer()
    with tr.span("tool:calc", "tool", expression="1+1"):
        pass
    ct = tr.to_chrome_trace()
    ev = ct["traceEvents"][0]
    assert ev["ph"] == "X" and ev["cat"] == "tool" and ev["args"]["expression"] == "1+1" and ev["dur"] >= 0
    p = tr.save(tmp_path / "t.json")
    assert json.loads(p.read_text())["spans"][0]["kind"] == "tool"
    assert (tmp_path / "t.chrome.json").exists()


def test_invalid_kind_rejected_and_reset():
    tr = Tracer()
    with pytest.raises(ValueError):
        with tr.span("x", "nope"):
            pass
    tr.reset()
    assert tr.spans == [] and tr.breakdown()["_wall_s"] == 0.0
