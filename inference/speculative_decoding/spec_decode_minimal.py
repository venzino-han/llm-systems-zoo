"""Minimal speculative decoding on top of transformers: draft proposes, target verifies.

The algorithm (Leviathan et al. 2023, Chen et al. 2023):

    repeat until done:
        1. the small DRAFT model autoregressively proposes K tokens d_1..d_K  (K small forwards)
        2. the TARGET model runs ONE forward over [prefix, d_1..d_K] and yields K+1 distributions
           p_1..p_{K+1}
        3. accept d_i left to right while it is "what the target would have produced";
           on the first rejection, emit a corrected token from the target and stop;
           if all K are accepted, emit one bonus token from p_{K+1}.

Two acceptance rules are implemented:

* greedy: accept d_i iff d_i == argmax p_i. Output is *bit-identical* to the target's own greedy
  decoding (asserted by `--check-equivalence`), so any speed-up is free of quality change.
* sampling: draft samples d_i ~ q_i; accept with probability min(1, p_i(d_i)/q_i(d_i)); on
  rejection sample from norm(max(0, p_i - q_i)). The resulting token distribution equals the
  target's (the rejection-sampling guarantee), checked in tests/test_l08_spec_decode.py.

KV cache handling: both models keep a `DynamicCache`. After verification we call `cache.crop(-n)`
to roll both caches back to the accepted length. Rolling back is exact (the cache entries for
accepted tokens are unchanged) and cheaper than recomputation; the alternative of recomputing the
prefix each step would hide the whole speed-up behind prefill cost.

Educational implementation: batch size 1, no tree attention, no CUDA graphs. Production engines
(vLLM, TensorRT-LLM, SGLang) batch the verification across requests and overlap draft and target
work; see bench_spec_decode.py for the vLLM numbers.

    .venv/bin/python inference/speculative_decoding/spec_decode_minimal.py --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import torch

from inference.quantization.common import wait_for_free_memory
from llm_systems_zoo import Run
from llm_systems_zoo.registry import MODELS


@dataclass
class SpecStats:
    target_forwards: int = 0
    draft_forwards: int = 0
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0
    accepted_per_position: list[int] = field(default_factory=list)  # accepted[i]: draft slot i accepted
    reached_per_position: list[int] = field(default_factory=list)

    def record(self, n_accepted: int, k: int) -> None:
        while len(self.accepted_per_position) < k:
            self.accepted_per_position.append(0)
            self.reached_per_position.append(0)
        for i in range(k):
            if i <= n_accepted:
                self.reached_per_position[i] += 1  # slot i was evaluated
            if i < n_accepted:
                self.accepted_per_position[i] += 1

    @property
    def acceptance_rate(self) -> float | None:
        return self.accepted / self.proposed if self.proposed else None

    @property
    def tokens_per_target_forward(self) -> float | None:
        return self.emitted / self.target_forwards if self.target_forwards else None

    @property
    def acceptance_rate_per_position(self) -> list[float | None]:
        return [a / r if r else None for a, r in zip(self.accepted_per_position, self.reached_per_position)]

    def summary(self) -> dict:
        return {
            "target_forwards": self.target_forwards,
            "draft_forwards": self.draft_forwards,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "emitted": self.emitted,
            "acceptance_rate": self.acceptance_rate,
            "tokens_per_target_forward": self.tokens_per_target_forward,
            "acceptance_rate_per_position": self.acceptance_rate_per_position,
        }



def _drop_tail(cache, n: int) -> None:
    """Remove the last ``n`` positions from a DynamicCache (no-op for ``n == 0``)."""
    if n > 0:
        cache.crop(-n)

def _new_cache():
    from transformers import DynamicCache

    return DynamicCache()


def _forward(model, ids: torch.Tensor, cache):
    """Run `ids` (1 x n) through the model, appending to `cache`; returns logits for each position."""
    past = cache.get_seq_length()
    pos = torch.arange(past, past + ids.shape[1], device=ids.device)[None]
    out = model(input_ids=ids, past_key_values=cache, position_ids=pos, use_cache=True)
    return out.logits[0]


def _sample(probs: torch.Tensor, generator) -> int:
    return int(torch.multinomial(probs.cpu() if generator is not None else probs, 1, generator=generator).item())


def verify_sampled(p: torch.Tensor, q: torch.Tensor, tok: int, generator=None) -> tuple[bool, int | None]:
    """Rejection-sampling step for one proposal `tok ~ q` against the target distribution `p`.

    Accept with probability min(1, p(tok)/q(tok)); on rejection return a token drawn from the
    residual norm(max(0, p - q)). Marginally, the emitted token is distributed exactly as p.
    """
    r = torch.rand((), generator=generator).item()
    if r < min(1.0, (p[tok] / q[tok]).item()):
        return True, None
    resid = torch.clamp(p - q, min=0)
    total = resid.sum()
    resid = resid / total if total > 0 else p
    return False, _sample(resid, generator)


@torch.no_grad()
def speculative_generate(
    target,
    draft,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    k: int = 4,
    mode: str = "greedy",
    temperature: float = 1.0,
    eos_token_id: int | None = None,
    generator=None,
) -> tuple[list[int], SpecStats]:
    """Returns (new_token_ids, stats). `input_ids` is 1 x L on the models' device.

    Invariant at the top of every round: `last_token` is the most recent token of the sequence,
    the target cache holds exactly the tokens *before* it, and the draft cache holds the tokens
    before it except those listed in `d_pending` (a proposal the draft never consumed because the
    whole block was accepted).
    """
    assert mode in ("greedy", "sample")
    stats = SpecStats()
    t_cache, d_cache = _new_cache(), _new_cache()
    device = input_ids.device
    if input_ids.shape[1] > 1:
        _forward(target, input_ids[:, :-1], t_cache)
        _forward(draft, input_ids[:, :-1], d_cache)
        stats.target_forwards += 1
    last_token = int(input_ids[0, -1])
    d_pending: list[int] = []
    generated: list[int] = []

    def dist(logits):
        return torch.softmax(logits.float() / max(temperature, 1e-5), dim=-1)

    while len(generated) < max_new_tokens:
        k_eff = min(k, max_new_tokens - len(generated))
        # ---- 1. draft proposes k_eff tokens ---------------------------------------------------
        proposals: list[int] = []
        q_dists: list[torch.Tensor | None] = []
        d_in = torch.tensor([[*d_pending, last_token]], device=device)
        for _ in range(k_eff):
            d_logits = _forward(draft, d_in, d_cache)[-1]
            stats.draft_forwards += 1
            if mode == "greedy":
                tok = int(d_logits.argmax().item())
                q_dists.append(None)
            else:
                q = dist(d_logits)
                tok = _sample(q, generator)
                q_dists.append(q)
            proposals.append(tok)
            d_in = torch.tensor([[tok]], device=device)
        stats.proposed += k_eff
        # ---- 2. target verifies all proposals in one forward ---------------------------------
        # logits[i] is the target's distribution for the token after position i, so logits[0]
        # judges proposals[0], ..., logits[k_eff-1] judges proposals[k_eff-1], logits[k_eff] is the
        # bonus distribution used when everything is accepted.
        t_all = _forward(target, torch.tensor([[last_token, *proposals]], device=device), t_cache)
        stats.target_forwards += 1
        # ---- 3. accept / reject left to right ---------------------------------------------------
        n_acc = 0
        corrected: int | None = None
        for i, tok in enumerate(proposals):
            if mode == "greedy":
                t_tok = int(t_all[i].argmax().item())
                if t_tok == tok:
                    n_acc += 1
                    continue
                corrected = t_tok
                break
            accepted, corrected = verify_sampled(dist(t_all[i]), q_dists[i], tok, generator)
            if accepted:
                n_acc += 1
                continue
            break
        stats.accepted += n_acc
        stats.record(n_acc, k_eff)
        if corrected is None:  # every proposal accepted -> bonus token for free
            corrected = int(t_all[k_eff].argmax().item()) if mode == "greedy" else _sample(dist(t_all[k_eff]), generator)
        new_tokens = [*proposals[:n_acc], corrected][: max_new_tokens - len(generated)]
        generated.extend(new_tokens)
        stats.emitted += len(new_tokens)
        # ---- 4. roll the caches back to the accepted prefix ------------------------------------
        # target saw last_token + k_eff proposals; keep last_token + the n_acc accepted ones.
        # Negative crop ("drop the last n entries") is the form supported by every transformers
        # release we test against; the positive "keep n" form was removed in 5.20.
        _drop_tail(t_cache, k_eff - n_acc)
        # draft saw (pending) + last_token + proposals[:k_eff-1]; keep up to the accepted ones.
        _drop_tail(d_cache, (k_eff - 1) - min(n_acc, k_eff - 1))
        d_pending = [proposals[k_eff - 1]] if n_acc == k_eff else []
        last_token = generated[-1]
        if eos_token_id is not None and eos_token_id in new_tokens:
            generated = generated[: len(generated) - len(new_tokens) + new_tokens.index(eos_token_id) + 1]
            break
    return generated, stats


@torch.no_grad()
def plain_generate(model, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None = None) -> list[int]:
    """Target-only greedy decoding through the same cache path (the baseline the speed-up is against)."""
    cache = _new_cache()
    logits = _forward(model, input_ids, cache)[-1]
    out: list[int] = []
    while len(out) < max_new_tokens:
        tok = int(logits.argmax().item())
        out.append(tok)
        if eos_token_id is not None and tok == eos_token_id:
            break
        logits = _forward(model, torch.tensor([[tok]], device=input_ids.device), cache)[-1]
    return out


def load_models(target_id: str, draft_id: str, device: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(target_id)
    target = AutoModelForCausalLM.from_pretrained(target_id, dtype=dtype).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(draft_id, dtype=dtype).to(device).eval()
    assert target.config.vocab_size == draft.config.vocab_size or tok.vocab_size <= min(target.config.vocab_size, draft.config.vocab_size), "draft and target must share a tokenizer"
    return target, draft, tok


DEFAULT_PROMPTS = [
    "Explain how a hash map works and when you would prefer a balanced tree instead.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "List the steps to make a cup of pour-over coffee, with timings.",
    "What are the trade-offs between tensor parallelism and pipeline parallelism for LLM inference?",
    "Summarise the plot of Romeo and Juliet in five sentences.",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default=MODELS["medium"].hf_id)
    p.add_argument("--draft", default=MODELS["draft"].hf_id)
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--n-prompts", type=int, default=len(DEFAULT_PROMPTS))
    p.add_argument("--check-equivalence", action="store_true", help="assert greedy spec output == target greedy output")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-free-gib", type=float, default=0.0, help="wait until this much GPU memory is free (shared GPUs)")
    args = p.parse_args()
    wait_for_free_memory(args.min_free_gib)
    if args.smoke:
        args.k, args.max_new_tokens, args.n_prompts, args.check_equivalence = [4], 48, 2, True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = vars(args).copy()
    with Run("L08_inference_optimization", "spec_decode_minimal", cfg, model={"target": args.target, "draft": args.draft, "dtype": "bfloat16"},
             workload={"prompts": "6 fixed prompts (see DEFAULT_PROMPTS)", "max_new_tokens": args.max_new_tokens, "mode": args.mode},
             execution={"label": "pilot, shared GPU", "batch": 1}) as run:
        target, draft, tok = load_models(args.target, args.draft, device)
        prompts = DEFAULT_PROMPTS[: args.n_prompts]
        enc = [tok.apply_chat_template([{"role": "user", "content": pr}], add_generation_prompt=True, return_tensors="pt").to(device) for pr in prompts]
        enc = [e if isinstance(e, torch.Tensor) else e["input_ids"] for e in enc]
        eos = tok.eos_token_id
        # baseline: target-only greedy
        base_tokens, base_time = [], 0.0
        for ids in enc:
            plain_generate(target, ids, 8, eos)  # warm-up
            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.perf_counter()
            out = plain_generate(target, ids, args.max_new_tokens, eos)
            torch.cuda.synchronize() if device == "cuda" else None
            base_time += time.perf_counter() - t0
            base_tokens.append(out)
        n_base = sum(len(o) for o in base_tokens)
        baseline = {"tokens": n_base, "wall_s": base_time, "ms_per_token": 1000 * base_time / n_base}
        run.log_metric(config="target_only", **baseline)
        results = []
        gen = torch.Generator().manual_seed(args.seed)
        for k in args.k:
            tot_time, tot_tokens, agg = 0.0, 0, SpecStats()
            equal = []
            for ids, ref in zip(enc, base_tokens):
                torch.cuda.synchronize() if device == "cuda" else None
                t0 = time.perf_counter()
                out, st = speculative_generate(target, draft, ids, args.max_new_tokens, k=k, mode=args.mode, temperature=args.temperature, eos_token_id=eos, generator=gen)
                torch.cuda.synchronize() if device == "cuda" else None
                tot_time += time.perf_counter() - t0
                tot_tokens += len(out)
                for f in ("target_forwards", "draft_forwards", "proposed", "accepted", "emitted"):
                    setattr(agg, f, getattr(agg, f) + getattr(st, f))
                for i, (a, r) in enumerate(zip(st.accepted_per_position, st.reached_per_position)):
                    while len(agg.accepted_per_position) <= i:
                        agg.accepted_per_position.append(0)
                        agg.reached_per_position.append(0)
                    agg.accepted_per_position[i] += a
                    agg.reached_per_position[i] += r
                equal.append(out == ref)
            row = {"k": k, "mode": args.mode, "tokens": tot_tokens, "wall_s": tot_time, "ms_per_token": 1000 * tot_time / tot_tokens,
                   "speedup_vs_target_only": baseline["ms_per_token"] / (1000 * tot_time / tot_tokens), "equal_to_target_greedy": sum(equal) / len(equal), **agg.summary()}
            if args.check_equivalence and args.mode == "greedy":
                assert all(equal), f"k={k}: speculative output differs from target greedy output on {len(equal) - sum(equal)} prompt(s)"
            run.log_metric(config=f"spec_k{k}", **{kk: vv for kk, vv in row.items() if not isinstance(vv, list)})
            results.append(row)
            print(json.dumps(row))
        summary = {"device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "target": args.target, "draft": args.draft,
                   "baseline_target_only": baseline, "speculative": results, "label": "pilot, shared GPU"}
        run.save_summary(summary)
        print("run_dir", run.dir)
        print(json.dumps(baseline))


if __name__ == "__main__":
    main()
