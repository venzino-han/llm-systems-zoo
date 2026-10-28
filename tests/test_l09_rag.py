from __future__ import annotations

import math

from applications.rag.evaluate import (
    bootstrap_ci,
    exact_match,
    f1_score,
    normalize_answer,
    recall_at_k,
)
from applications.rag.pipeline import SYSTEM_PROMPT, build_messages


def test_normalisation_and_exact_match():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert exact_match("Paris.", "paris") == 1.0
    assert exact_match("in Paris", "paris") == 0.0


def test_f1_partial_overlap():
    assert math.isclose(f1_score("Abraham Lincoln", "Lincoln"), 2 * 0.5 * 1 / 1.5)
    assert f1_score("", "x") == 0.0 and f1_score("", "") == 1.0


def test_recall_at_k():
    assert recall_at_k("1838", ["founded in 1838 by", "other"]) == 1.0
    assert recall_at_k("1838", ["nothing here"]) == 0.0


def test_bootstrap_ci_brackets_mean():
    lo, hi = bootstrap_ci([0, 1, 1, 1, 0, 1, 0, 1, 1, 1], n_boot=200)
    assert lo <= 0.7 <= hi and 0 <= lo and hi <= 1
    assert all(math.isnan(v) for v in bootstrap_ci([]))


def test_static_system_prompt_is_first_for_prefix_caching():
    m = build_messages("Who?", ["p1", "p2"])
    assert m[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "[1] p1" in m[1]["content"] and m[1]["content"].endswith("Answer:")
