"""Verifiable rewards for GSM8K-style math (L06).

A verifiable reward is a function of (completion, gold) with no learned judge: cheap, free of
reward hacking through a model, and deterministic. Its weakness is coverage: it only scores
what it can parse, so a format reward is added to teach the parseable format first.
"""

from __future__ import annotations

import re
from typing import Any

SYSTEM_PROMPT = (
    "Solve the problem step by step. After your reasoning, write the final numeric answer on its own "
    "line in the form '#### <number>'."
)

_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_HASH_ANSWER = re.compile(r"####\s*(" + _NUM + ")")
_ANY_NUM = re.compile(_NUM)


def normalise_number(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v.is_integer() else f"{v:g}"


def gold_answer(gsm8k_answer: str) -> str:
    """GSM8K gold solutions end with '#### <answer>'."""
    m = _HASH_ANSWER.search(gsm8k_answer)
    if not m:
        raise ValueError(f"no '####' answer in: {gsm8k_answer[-80:]!r}")
    out = normalise_number(m.group(1))
    assert out is not None
    return out


def extract_answer(completion: str) -> str | None:
    """Prefer the '#### n' form; fall back to the last number in the text."""
    m = _HASH_ANSWER.findall(completion)
    if m:
        return normalise_number(m[-1])
    nums = _ANY_NUM.findall(completion)
    return normalise_number(nums[-1]) if nums else None


def has_answer_format(completion: str) -> bool:
    return _HASH_ANSWER.search(completion) is not None


def _text(completion: Any) -> str:
    """TRL passes either a string or a list of chat messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return "".join(str(m.get("content", "")) for m in completion)
    return str(completion)


# --- TRL reward function signature: (prompts, completions, **dataset_columns) -> list[float] ---
def correctness_reward(prompts=None, completions=None, answer=None, **kwargs) -> list[float]:
    golds = answer if answer is not None else kwargs.get("gold")
    return [1.0 if extract_answer(_text(c)) == g else 0.0 for c, g in zip(completions, golds)]


def format_reward(prompts=None, completions=None, **kwargs) -> list[float]:
    return [0.2 if has_answer_format(_text(c)) else 0.0 for c in completions]


def length_penalty(prompts=None, completions=None, max_chars: int = 1500, **kwargs) -> list[float]:
    """Mild penalty for run-away completions (keeps rollouts short → cheaper generation)."""
    return [0.0 if len(_text(c)) <= max_chars else -0.2 for c in completions]


REWARD_FUNCS = {"correctness": correctness_reward, "format": format_reward, "length": length_penalty}
