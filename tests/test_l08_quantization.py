"""CPU tests for the perplexity / quantisation tooling (tiny random model, no downloads)."""

from __future__ import annotations

import logging
import math

import pytest

from inference.quantization.bench_quant import table
from inference.quantization.common import VllmLogCapture, resolve_checkpoint
from inference.quantization.eval_perplexity import window_bounds


def test_window_bounds_non_overlapping_skips_window_boundary_tokens():
    # Token 0 of every window has no context inside the window, so with stride == max_length the
    # boundary tokens (4 and 8 here) are never scored. That is the standard HF behaviour, and the
    # reason the sliding-window variant exists.
    wins = window_bounds(seq_len=10, max_length=4, stride=4)
    assert wins == [(0, 4, 1), (4, 8, 5), (8, 10, 9)]
    scored = [j for b, e, first in wins for j in range(first, e)]
    assert scored == [1, 2, 3, 5, 6, 7, 9]


def test_window_bounds_sliding_scores_each_token_once_with_more_context():
    wins = window_bounds(seq_len=10, max_length=6, stride=3)
    scored = [j for b, e, first in wins for j in range(first, e)]
    assert scored == list(range(1, 10))
    assert wins[-1][1] == 10


def test_perplexity_matches_direct_computation():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    from inference.quantization.eval_perplexity import nll_from_logits

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)).eval()
    ids = torch.randint(0, 50, (23,))
    # direct: one forward over the whole sequence
    with torch.no_grad():
        logits = model(ids[None]).logits[0]
    direct = torch.nn.functional.cross_entropy(logits[:-1].float(), ids[1:], reduction="sum").item()
    s, n = nll_from_logits(logits, ids, 1)
    assert math.isclose(s, direct, rel_tol=1e-5) and n == 22
    # windowed, non-overlapping: scores the same 22 tokens, windows 2+ have less context
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for b, e, first in window_bounds(23, 8, 8):
            w = ids[b:e]
            s, n = nll_from_logits(model(w[None]).logits[0], w, first - b)
            tot += s
            cnt += n
    assert cnt == 20 and tot > 0  # 22 minus the two window-boundary tokens (8 and 16)


def test_resolve_checkpoint():
    m = "Qwen/Qwen2.5-0.5B-Instruct"
    assert resolve_checkpoint(m, "bf16") == m
    assert resolve_checkpoint(m, "awq") == m + "-AWQ"
    assert resolve_checkpoint(m, "gptq") == m + "-GPTQ-Int4"
    assert resolve_checkpoint(m + "-AWQ", "awq") == m + "-AWQ"


def test_vllm_log_capture_parses_memory_lines():
    cap = VllmLogCapture()
    log = logging.getLogger("vllm.test")
    log.setLevel(logging.INFO)
    cap.install()
    try:
        log.info("Model loading took 0.9260 GiB and 1.2 seconds")
        log.info("GPU KV cache size: 123,456 tokens")
        log.info("Maximum concurrency for 2,048 tokens per request: 60.28x")
        log.info("Graph capturing finished in 3 secs, took 0.12 GiB")
    finally:
        cap.remove()
    assert cap.values["model_weights_gib"] == 0.926
    assert cap.values["kv_cache_tokens"] == 123456
    assert cap.values["max_concurrency_tokens_per_request"] == 2048 and cap.values["max_concurrency_x"] == 60.28
    assert cap.values["cuda_graph_capture_s"] == 3


def test_bench_quant_table_keeps_failures():
    rows = table([
        {"status": "ok", "engine": "hf", "load": "bf16", "checkpoint": "m", "weights_gib": 1.0,
         "perplexity": {"ppl": 10.0, "scored_tokens": 100, "prefill_tokens_per_s": 5.0},
         "decode": [{"batch": 1, "output_tokens_per_s": 50.0, "ms_per_step": 20.0}]},
        {"status": "failed", "engine": "hf", "load": "awq", "checkpoint": "m-AWQ", "error": "ImportError: autoawq"},
    ])
    assert rows[0]["ppl"] == 10.0 and rows[0]["decode_tok_s_b1"] == 50.0
    assert rows[1]["status"] == "failed" and "autoawq" in rows[1]["error"]
