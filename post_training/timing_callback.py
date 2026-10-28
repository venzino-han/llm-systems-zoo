"""Phase decomposition of TRL trainers (L06).

TRL does not expose "rollout / reward / learner" hooks, so this module wraps the trainer's own
methods at the *instance* level (no monkey-patching of the class) and attributes their wall
time to named phases with `PhaseTimer`. What is wrapped, per trainer, for TRL 1.9.x:

GRPOTrainer
  _generate_and_score_completions  → "rollout_and_score"   (outer: generation + rewards + logps)
    _generate                      → "rollout"             (HF generate or vLLM; nested)
    _calculate_rewards             → "reward"              (nested)
  compute_loss                     → "learner_forward_loss"(policy forward + loss; nested in training_step)
  training_step                    → "train_step"          (outer; includes prepare_inputs → generation)
  vllm_generation.sync_weights     → "weight_sync"         (only when use_vllm)
  derived: learner_backward = train_step - rollout_and_score - learner_forward_loss
           optimizer        = on_pre_optimizer_step → on_optimizer_step callback window

DPOTrainer
  _precompute_ref_logps            → "ref_precompute"      (one-off pass when precompute_ref_log_probs=True)
  compute_ref_log_probs            → "ref_forward"         (only called from the precompute pass in TRL 1.9)
  compute_loss                     → "policy_forward_loss" (in the default path this *includes* the inline
                                                            reference forward: TRL 1.9 computes it inside
                                                            `_compute_loss` under `use_adapter`, not via a
                                                            wrappable method)
  training_step                    → "train_step"
  derived: reference share per step = policy_forward(default) − policy_forward(--precompute-ref),
           i.e. an A/B of two runs rather than a wrapper.

SFTTrainer
  compute_loss                     → "forward_loss";  training_step → "train_step"

Nested phases are *not* subtracted automatically; `phase_breakdown()` derives the exclusive
shares so the report can show a stacked bar that sums to the step time.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from transformers import TrainerCallback

from llm_systems_zoo import PhaseTimer

WRAP_PLAN: dict[str, dict[str, str]] = {
    "GRPOTrainer": {
        "_generate_and_score_completions": "rollout_and_score",
        "_generate": "rollout",
        "_calculate_rewards": "reward",
        "compute_loss": "learner_forward_loss",
        "training_step": "train_step",
    },
    "DPOTrainer": {
        "compute_ref_log_probs": "ref_forward",
        "_precompute_ref_logps": "ref_precompute",
        "compute_loss": "policy_forward_loss",
        "training_step": "train_step",
    },
    "SFTTrainer": {"compute_loss": "forward_loss", "training_step": "train_step"},
}

EXCLUSIVE_RULES: dict[str, dict[str, list[str]]] = {
    # phase: [nested phases to subtract]
    "GRPOTrainer": {"rollout_and_score": ["rollout", "reward"], "train_step": ["rollout_and_score", "learner_forward_loss"]},
    "DPOTrainer": {"policy_forward_loss": ["ref_forward"], "train_step": ["policy_forward_loss"]},
    "SFTTrainer": {"train_step": ["forward_loss"]},
}

RENAME_EXCLUSIVE = {
    "GRPOTrainer": {"rollout_and_score": "score_other", "train_step": "learner_backward"},
    "DPOTrainer": {"policy_forward_loss": "policy_forward", "train_step": "backward"},
    "SFTTrainer": {"train_step": "backward"},
}


def _wrap(obj: Any, method: str, timer: PhaseTimer, phase: str) -> bool:
    fn: Callable | None = getattr(obj, method, None)
    if fn is None or not callable(fn):
        return False

    @functools.wraps(fn)
    def timed(*args, **kwargs):
        with timer.phase(phase):
            return fn(*args, **kwargs)

    setattr(obj, method, timed)
    return True


def instrument_trainer(trainer: Any, timer: PhaseTimer, extra: dict[str, str] | None = None) -> list[str]:
    """Wrap the trainer's phase methods. Returns the phases actually wrapped."""
    kind = type(trainer).__name__
    plan = dict(WRAP_PLAN.get(kind, {}))
    plan.update(extra or {})
    wrapped = [phase for method, phase in plan.items() if _wrap(trainer, method, timer, phase)]
    vg = getattr(trainer, "vllm_generation", None)
    if vg is not None and _wrap(vg, "sync_weights", timer, "weight_sync"):
        wrapped.append("weight_sync")
    elif _wrap(trainer, "_move_model_to_vllm", timer, "weight_sync"):
        wrapped.append("weight_sync")
    return wrapped


def phase_breakdown(totals: dict[str, float], kind: str) -> dict[str, float]:
    """Exclusive seconds per phase (nested children subtracted), renamed to what they measure."""
    ex = dict(totals)
    for phase, children in EXCLUSIVE_RULES.get(kind, {}).items():
        if phase in ex:
            ex[phase] = max(ex[phase] - sum(totals.get(c, 0.0) for c in children), 0.0)
    out = {}
    for k, v in ex.items():
        out[RENAME_EXCLUSIVE.get(kind, {}).get(k, k)] = v
    return out


class PhaseTimingCallback(TrainerCallback):
    """Records per-step exclusive phase times and the optimizer window; logs through `log_fn`."""

    def __init__(self, timer: PhaseTimer, kind: str, log_fn: Callable[..., None] | None = None) -> None:
        self.timer = timer
        self.kind = kind
        self.log_fn = log_fn
        self.step_rows: list[dict[str, Any]] = []
        self._last_totals: dict[str, float] = {}
        self._t_opt: float | None = None
        self._t_step0 = time.perf_counter()

    def on_pre_optimizer_step(self, args, state, control, **kw):
        self._t_opt = time.perf_counter()

    def on_optimizer_step(self, args, state, control, **kw):
        if self._t_opt is not None:
            self.timer.add("optimizer", time.perf_counter() - self._t_opt)
            self._t_opt = None

    def on_step_end(self, args, state, control, **kw):
        totals = dict(self.timer.totals)
        delta = {k: totals[k] - self._last_totals.get(k, 0.0) for k in totals}
        self._last_totals = totals
        row = {"step": state.global_step, "wall_s": time.perf_counter() - self._t_step0, **phase_breakdown(delta, self.kind)}
        self._t_step0 = time.perf_counter()
        self.step_rows.append(row)
        if self.log_fn:
            self.log_fn(**row)

    def on_log(self, args, state, control, logs=None, **kw):
        if self.log_fn and logs:
            self.log_fn(step=state.global_step, **{f"trainer/{k}": v for k, v in logs.items() if isinstance(v, (int, float))})

    def summary(self) -> dict[str, Any]:
        rows = self.step_rows[1:] if len(self.step_rows) > 1 else self.step_rows  # drop the warm-up step
        if not rows:
            return {}
        keys = [k for k in rows[0] if k not in ("step", "wall_s")]
        mean = {k: sum(r.get(k, 0.0) for r in rows) / len(rows) for k in keys}
        wall = sum(r["wall_s"] for r in rows) / len(rows)
        return {
            "n_steps": len(rows),
            "mean_step_wall_s": wall,
            "mean_phase_s": mean,
            "phase_share": {k: (v / wall if wall else 0.0) for k, v in mean.items()},
            "unaccounted_share": max(1 - sum(mean.values()) / wall, 0.0) if wall else 0.0,
        }
