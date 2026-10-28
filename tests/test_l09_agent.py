from __future__ import annotations

from applications.agent.agent import Agent, Verifier, decide_support, parse_action, split_claims
from applications.agent.tools import calculator, lookup_date, render_tool_help
from applications.trace import Tracer


def test_parse_action_variants():
    assert parse_action('{"thought": "t", "action": "calculator", "args": {"expression": "1+1"}}')["action"] == "calculator"
    assert parse_action('Sure!\n```json\n{"action": "final", "answer": "42"}\n```')["answer"] == "42"
    assert parse_action('{"action": "x", "args": "bad"}')["args"] == {}
    assert parse_action("no json here") is None
    assert parse_action('{"foo": 1} {"action": "final", "answer": "a"}')["action"] == "final"


def test_calculator_safe():
    assert calculator("123*45 + 6") == "5541"
    assert calculator("(2+3)/2") == "2.5"
    assert calculator("__import__('os')").startswith("error")
    assert calculator("2**1000").startswith("error")
    assert lookup_date("2026-10-29") == "2026-10-29"
    assert "calculator(expression: string)" in render_tool_help()


def test_claim_support_decisions():
    assert split_claims("A is B. C is D! E?") == ["A is B.", "C is D!", "E?"]
    assert decide_support("x", [0.5, -1.0], [], threshold=0.0)["reason"] == "passage"
    assert decide_support("The answer is 5541.", [-3.0], ["calculator -> 5541"], 0.0)["reason"] == "tool_observation"
    assert not decide_support("The answer is 5541.", [-3.0], ["calculator -> 12"], 0.0)["supported"]
    assert not decide_support("no numbers", [], [], 0.0)["supported"]
    v = Verifier(lambda claim, passages: [1.0 if ("sun" in p and "sun" in claim) else -1.0 for p in passages], 0.0)
    out = v.verify("The sun is hot. Cats bark.", ["the sun is a star", "dogs"], [])
    assert [c["supported"] for c in out] == [True, False]


class ScriptedGenerator:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, messages, max_new_tokens=None, tools=None):
        self.calls += 1
        text = self.replies.pop(0) if self.replies else '{"action": "final", "answer": "fallback"}'
        return {"text": text, "prompt_tokens": 10, "output_tokens": 5}


def test_agent_loop_uses_tool_then_finishes_and_verifies():
    gen = ScriptedGenerator(['{"thought": "calc", "action": "calculator", "args": {"expression": "123*45+6"}}', '{"thought": "done", "action": "final", "answer": "The result is 5541."}'])
    agent = Agent(gen, {"calculator": calculator}, Tracer(), max_steps=3, verifier=Verifier(None, 0.0), verify_mode="flag")
    out = agent.run("What is 123*45+6?")
    assert out["n_tool_calls"] == 1 and out["n_model_calls"] == 2
    assert out["steps"][0]["observation"] == "5541" and out["answer"] == "The result is 5541."
    assert out["verification"][0]["supported"] and out["latency_s"]["tool"] >= 0


def test_agent_flags_or_removes_unsupported_claims_and_forces_final():
    gen = ScriptedGenerator(['{"action": "final", "answer": "Paris is the capital. It has 99 bridges."}'])
    agent = Agent(gen, {"calculator": calculator}, Tracer(), max_steps=2, verifier=Verifier(None, 0.0), verify_mode="remove")
    assert agent.run("q")["answer"] == "unknown"  # no evidence at all -> every claim removed
    gen = ScriptedGenerator(['{"action": "final", "answer": "Paris is the capital."}'])
    agent = Agent(gen, {"calculator": calculator}, Tracer(), max_steps=2, verifier=Verifier(None, 0.0), verify_mode="flag")
    assert "[unverified" in agent.run("q")["answer"]
    gen = ScriptedGenerator(['{"action": "lookup_date", "args": {}}'] * 5)  # never finishes on its own
    agent = Agent(gen, {"lookup_date": lambda: "2026-01-01"}, Tracer(), max_steps=2)
    out = agent.run("date?")
    assert out["n_tool_calls"] == 2 and out["steps"][-1]["action"] == "final"


def test_parse_action_salvages_truncated_json_and_calculator_tolerates_suffixes():
    t = parse_action('{"thought": "long reasoning that never ends', )
    assert t is None
    t = parse_action('{"thought": "x", "action": "final", "answer": "Paris is the')
    assert t["action"] == "final" and t["answer"] == "Paris is the" and t["salvaged"]
    t = parse_action('{"thought": "x", "action": "calculator", "args": {"expression": "1+2"')
    assert t["action"] == "calculator" and t["args"] == {"expression": "1+2"} and t["salvaged"]
    assert calculator("What is 876 * 51 + 777 = ?") == "45453"
    assert calculator("2^3") == "8"


def test_yes_no_scoring():
    from applications.agent.evaluate import score

    assert score({"kind": "qa", "answer": "yes"}, "Yes, foie gras is often made from ducks.")["success"] == 1.0
    assert score({"kind": "qa", "answer": "Yes."}, "No, it is not.")["success"] == 0.0
    assert score({"kind": "qa", "answer": "Kazakhstan"}, "In Kazakhstan.")["success"] == 1.0
    assert score({"kind": "arith", "answer": "42"}, "The result is 42.")["success"] == 1.0
