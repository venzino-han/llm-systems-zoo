"""L06 tests: reward parsing, timing instrumentation bookkeeping, bootstrap CI. CPU, no network."""

from __future__ import annotations

import math
import time

import pytest

from llm_systems_zoo import PhaseTimer
from post_training.common import bootstrap_ci
from post_training.rewards import (
    correctness_reward,
    extract_answer,
    format_reward,
    gold_answer,
    has_answer_format,
    length_penalty,
    normalise_number,
)
from post_training.timing_callback import PhaseTimingCallback, instrument_trainer, phase_breakdown


# ---- rewards --------------------------------------------------------------------------------
def test_gold_answer_parses_gsm8k_format():
    assert gold_answer("She has 3 apples.\n#### 3") == "3"
    assert gold_answer("Total: $1,234.50\n#### 1,234.50") == "1234.5"
    with pytest.raises(ValueError):
        gold_answer("no marker here")


@pytest.mark.parametrize("text,expected", [
    ("so the answer is #### 42", "42"),
    ("#### 7\nActually #### 8", "8"),  # last marked answer wins
    ("Step 1: 3+4=7. Final: 7.", "7"),  # fallback: last number
    ("The total is $1,200.", "1200"),
    ("no numbers at all", None),
    ("#### -3.0", "-3"),
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_normalise_number_handles_junk():
    assert normalise_number("1,000") == "1000" and normalise_number("abc") is None and normalise_number("") is None


def test_reward_functions_with_strings_and_chat_messages():
    comps = ["#### 5", [{"role": "assistant", "content": "I think #### 6"}], "5"]
    golds = ["5", "6", "5"]
    assert correctness_reward(prompts=None, completions=comps, answer=golds) == [1.0, 1.0, 1.0]
    assert format_reward(prompts=None, completions=comps) == [0.2, 0.2, 0.0]
    assert has_answer_format("#### 12") and not has_answer_format("12")
    assert length_penalty(prompts=None, completions=["x" * 10, "y" * 2000]) == [0.0, -0.2]


def test_correctness_reward_rejects_wrong_answers():
    assert correctness_reward(completions=["#### 4", "#### 44"], answer=["44", "44"]) == [0.0, 1.0]


# ---- timing instrumentation -----------------------------------------------------------------
class FakeGRPOTrainer:
    def __init__(self):
        self.vllm_generation = None

    def _generate(self, prompts):
        time.sleep(0.01)
        return prompts

    def _calculate_rewards(self, *a):
        time.sleep(0.005)
        return [1.0]

    def _generate_and_score_completions(self, inputs):
        self._generate(inputs)
        self._calculate_rewards(inputs)
        time.sleep(0.002)  # the "other" scoring work (logps, advantages)
        return inputs

    def compute_loss(self, *a, **k):
        time.sleep(0.004)
        return 0.0

    def training_step(self, inputs):
        self._generate_and_score_completions(inputs)
        self.compute_loss(inputs)
        time.sleep(0.003)  # backward
        return 0.0


FakeGRPOTrainer.__name__ = "GRPOTrainer"


class _State:
    global_step = 0


def test_instrument_trainer_wraps_methods_and_breakdown_is_exclusive():
    t = FakeGRPOTrainer()
    timer = PhaseTimer()
    wrapped = instrument_trainer(t, timer)
    assert {"rollout", "reward", "rollout_and_score", "learner_forward_loss", "train_step"} <= set(wrapped)
    cb = PhaseTimingCallback(timer, "GRPOTrainer")
    state = _State()
    for step in range(1, 4):
        t.training_step(["p"])
        cb.on_pre_optimizer_step(None, state, None)
        time.sleep(0.001)
        cb.on_optimizer_step(None, state, None)
        state.global_step = step
        cb.on_step_end(None, state, None)
    rows = cb.step_rows
    assert len(rows) == 3
    r = rows[-1]
    # exclusive phases: rollout ≈ 10ms, reward ≈ 5ms, score_other ≈ 2ms, learner_forward ≈ 4ms, backward ≈ 3ms
    assert r["rollout"] >= 0.009 and r["reward"] >= 0.004 and r["learner_forward_loss"] >= 0.003
    assert 0.0 <= r["score_other"] < 0.01 and 0.0 <= r["learner_backward"] < 0.01
    assert r["optimizer"] >= 0.001
    s = cb.summary()
    assert s["n_steps"] == 2 and abs(sum(s["phase_share"].values()) + s["unaccounted_share"] - 1.0) < 1e-6
    assert s["phase_share"]["rollout"] > s["phase_share"]["reward"]


def test_phase_breakdown_subtracts_children_and_renames():
    totals = {"rollout_and_score": 10.0, "rollout": 6.0, "reward": 1.0, "learner_forward_loss": 2.0, "train_step": 15.0}
    ex = phase_breakdown(totals, "GRPOTrainer")
    assert ex["score_other"] == 3.0 and ex["learner_backward"] == 3.0 and ex["rollout"] == 6.0
    dpo = phase_breakdown({"policy_forward_loss": 5.0, "ref_forward": 2.0, "train_step": 8.0}, "DPOTrainer")
    assert dpo["policy_forward"] == 3.0 and dpo["backward"] == 3.0 and dpo["ref_forward"] == 2.0


def test_instrument_unknown_trainer_wraps_nothing():
    class Other:
        pass

    assert instrument_trainer(Other(), PhaseTimer()) == []


# ---- statistics -----------------------------------------------------------------------------
def test_bootstrap_ci_brackets_mean():
    ci = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 0, 1, 1], n_boot=500)
    assert ci["n"] == 10 and math.isclose(ci["mean"], 0.7)
    assert ci["ci_low"] <= 0.7 <= ci["ci_high"] and ci["ci_low"] < ci["ci_high"]
    assert bootstrap_ci([])["n"] == 0
