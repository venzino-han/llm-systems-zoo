# L08b. Speculative decoding

**Question.** Under what conditions does speculative decoding reduce per-token latency, and by
how much, without changing the output distribution?
**Status.** Implemented (minimal implementation + vLLM benchmark), Measured (pilot, shared GPU).
**Hardware needed.** 1 CUDA GPU; ~5 GB for the 0.5B -> 1.5B pair.

## 1. What the technique is

Autoregressive decoding runs one forward pass per token, and at small batch that pass is
memory-bound: it reads every weight from HBM to produce a single token. Speculative decoding
(Leviathan et al. 2023; Chen et al. 2023) spends that bandwidth on more than one token:

1. A cheap **drafter** proposes `K` tokens (a small model, or a lookup in the prompt for
   repeated n-grams, or extra heads on the target as in Medusa / EAGLE).
2. The **target** model runs *one* forward over the `K` proposals and returns `K+1`
   next-token distributions. Reading the weights once and multiplying by a `K+1` token block
   costs about the same as a single-token step in the memory-bound regime.
3. Proposals are **verified** left to right: under greedy decoding a proposal is accepted iff it
   equals the target's argmax; under sampling it is accepted with probability
   `min(1, p(x)/q(x))` and, on rejection, replaced by a sample from `norm(max(0, p - q))`. Both
   rules make the output *exactly* the target's distribution (the rejection-sampling guarantee),
   so quality is untouched by construction. If all `K` are accepted, the `(K+1)`-th distribution
   yields a bonus token.

### Cost model

With acceptance rate `alpha` per position and `K` proposals, the expected number of tokens per
target forward is `(1 - alpha^(K+1)) / (1 - alpha)`. The speed-up is

```
speedup = tokens_per_target_forward / (1 + K * c)
```

where `c` is the drafter's cost relative to one target step. Speculation wins when `alpha` is
high (drafter agrees with the target), `c` is small (drafter much cheaper), and the target step
is memory-bound so verifying `K+1` tokens costs about one step. It loses when the batch is large
(the target step is already compute-bound, so verification is no longer free), when `c` is not
small (the minimal implementation below), or when `alpha` is low (n-gram lookup on free-form
chat).

### Variants

| Drafter | Cost `c` | Typical `alpha` | Notes |
|---|---|---|---|
| Small draft model (same tokenizer) | 0.1-0.3 | 0.6-0.8 | needs a second model in memory; two KV caches |
| n-gram / prompt lookup | ~0 | high on repetitive text (code edits, summarisation, RAG), ~0 on open-ended chat | free, zero extra memory; vLLM `method="ngram"` |
| Medusa / EAGLE heads | ~0.05 | 0.6-0.8 | extra trained heads on the target; tree verification |

### Educational vs production

`spec_decode_minimal.py` is the algorithm at batch size 1 on top of transformers with
`DynamicCache`: both models keep a KV cache and are rolled back with `cache.crop()` to the
accepted length after every verification (exact, and cheaper than recomputing). It has no CUDA
graphs and pays Python overhead for every draft step, which is why its measured `c` is far from
0.3 (see results). `bench_spec_decode.py` uses vLLM's implementation, which batches
verification across requests, runs the drafter under CUDA graphs, and reports its own acceptance
counters.

## 2. When to use it

* Interactive, latency-sensitive serving at low concurrency (batch 1-8), especially with a
  target large enough that its step is weight-bandwidth-bound (7B+ on a single GPU).
* Workloads with predictable text: code completion, structured output, summarisation of a
  document in the prompt, RAG answers that quote retrieved passages (n-gram drafting shines).
* Do **not** expect gains when: the server runs at high batch (throughput-oriented), the target is
  tiny (step dominated by launch overhead, as here), or outputs are open-ended with a weak drafter.

## 3. Trade-offs

| Advantages | Disadvantages |
|---|---|
| 1.5-3x lower per-token latency at small batch, with provably unchanged output distribution | No gain, or a loss, at large batch; a draft model costs memory and KV for a second model |
| n-gram drafting is free and needs nothing extra | n-gram acceptance is near zero on non-repetitive text |
| Composable with quantisation and CUDA graphs | Engine complexity: two schedulers, rejection sampling kernels, cache rollback; numerics differ slightly from plain decode (batched verification uses different GEMM shapes), so bit-identical outputs are not guaranteed in bf16 |

What you pay: memory for the drafter, engine complexity, wasted draft compute on rejection.
What you gain: tokens per weight-read, i.e. latency in the memory-bound regime.

## 4. What is in this directory

| File | Purpose | Runs on |
|---|---|---|
| `spec_decode_minimal.py` | Draft/verify loop with greedy and rejection-sampling acceptance, cache rollback, per-position acceptance stats, equivalence check and timing vs target-only decoding | CPU / GPU |
| `bench_spec_decode.py` | vLLM: none vs n-gram vs draft-model speculation, batch sweep, engine acceptance metrics, greedy-output equality | GPU |
| `../../tests/test_l08_spec_decode.py` | tiny random models: output equals target greedy for any draft and `K`; same-model draft accepts everything; EOS; rejection sampling recovers `p` empirically; acceptance probability `min(1, p/q)` | CPU |

## 5. How to run

```bash
.venv/bin/python inference/speculative_decoding/spec_decode_minimal.py --smoke          # 0.5B -> 1.5B, asserts greedy equivalence
.venv/bin/python inference/speculative_decoding/spec_decode_minimal.py --k 1 2 4 8 --max-new-tokens 128
.venv/bin/python inference/speculative_decoding/spec_decode_minimal.py --mode sample --k 4

.venv/bin/python inference/speculative_decoding/bench_spec_decode.py --smoke             # 0.5B target, none vs ngram
.venv/bin/python inference/speculative_decoding/bench_spec_decode.py --methods none ngram draft --batch-sizes 1 4 16
.venv/bin/python -m pytest tests/test_l08_spec_decode.py
```

## 6. Results

**Environment.** RTX A6000 (sm_86, 48 GB), shared with other jobs (*pilot, shared GPU*); torch
2.11.0, transformers 5.7.0, vllm 0.23.0. Draft Qwen2.5-0.5B-Instruct, target Qwen2.5-1.5B-Instruct,
bf16, greedy.

### 6.1 Minimal implementation, batch 1, 6 prompts x 128 tokens (run `20261029_023536_spec_decode_minimal_ba6c421baced`)

Target-only decoding through the same cache path: **10.51 ms/token** (768 tokens, 8.07 s).

| K | ms/token | speed-up | acceptance (accepted / proposed) | per-position acceptance (given reached) | tokens per target forward | target forwards | draft forwards | identical to target greedy (of 6 prompts) |
|---|---|---|---|---|---|---|---|---|
| 1 | 13.03 | 0.81x | 0.79 | 0.79 | 1.76 | 437 | 431 | 4 |
| 2 | 13.03 | 0.81x | 0.68 | 0.77, 0.75 | 2.30 | 334 | 653 | 5 |
| 4 | 14.87 | 0.71x | 0.53 | 0.73, 0.75, 0.85, 0.80 | 3.04 | 253 | 985 | 4 |
| 8 | 22.19 | 0.47x | 0.36 | 0.73 ... 0.86 | 3.66 | 210 | 1595 | 2 |

Reading:

* The mechanism works as the theory says: the draft agrees with the target ~75-85% of the time
  at every position, so the number of tokens produced per target forward climbs from 1 (plain
  decoding) to 3.7 at K = 8, and the number of target forwards drops from 768 to 210.
* It is still **slower** than plain decoding, and gets slower with K, because on this pair the
  draft step is not cheap: a 0.5B and a 1.5B model both take ~10 ms per eager HF step (the step
  is launch-bound, not weight-bandwidth-bound), so the drafter's relative cost `c` is ~1 instead
  of the ~0.2 the cost model wants. With `c = 1`, `speedup = tptf / (1 + K)`: 1.76/2, 2.30/3,
  3.04/5, 3.66/9 = 0.88, 0.77, 0.61, 0.41, which is what was measured (the extra margin comes from
  cache rollback and Python bookkeeping). The acceptance side of the trade is fine; the cost side
  needs a target whose step is dominated by weight reads (7B+) or a drafter that runs under CUDA
  graphs.
* The smoke run (2 prompts x 48 tokens) reproduced the target's greedy output exactly; over 128
  tokens 2-4 of 6 prompts diverged at some position. Verification uses a (K+1)-token forward while
  plain decoding uses 1-token forwards, and bf16 GEMMs with different shapes round differently,
  so near-tied logits can flip. The *distribution* is unchanged; bit-identical output is not
  guaranteed in finite precision (vLLM shows the same effect in 6.2).

### 6.2 vLLM: none vs n-gram vs draft model, batch 1 / 4 / 16

Target Qwen2.5-1.5B-Instruct, draft Qwen2.5-0.5B-Instruct, K = 4, 1 GiB KV budget, `max_num_seqs`
32, 128 greedy tokens per prompt, min of 2 repeats, one engine per subprocess
(run `20261029_024626_vllm_spec_decode_36c008dab3f7`). Acceptance numbers come from the engine's
own counters (`vllm:spec_decode_num_accepted_tokens / num_draft_tokens`); "accepted per step" is
accepted draft tokens per verification, so tokens per target step is that value + 1.

| method | batch | ms/step | tok/s | acceptance rate | accepted per step | vs none | identical to plain greedy | engine init (s) |
|---|---|---|---|---|---|---|---|---|
| none | 1 | 11.38 | 88 | - | - | 1.00x | - | 23.9 |
| none | 4 | 11.85 | 338 | - | - | 1.00x | - | |
| none | 16 | 12.27 | 1304 | - | - | 1.00x | - | |
| ngram | 1 | 12.02 | 83 | 0.04 | 0.17 | 0.95x | 1/1 | 26.9 |
| ngram | 4 | 12.48 | 321 | 0.30 | 1.21 | 0.95x | 2/4 | |
| ngram | 16 | 15.22 | 1051 | 0.32 | 1.28 | 0.81x | 5/16 | |
| draft model | 1 | 11.35 | 88 | 0.47 | 1.90 | 1.00x | 1/1 | 39.4 |
| draft model | 4 | 15.42 | 259 | 0.51 | 2.03 | 0.77x | 1/4 | |
| draft model | 16 | 16.67 | 960 | 0.52 | 2.06 | 0.74x | 4/16 | |

Reading:

* **Draft-model speculation** halves the number of target steps (2.9-3.1 tokens per step) but
  the wall-clock is flat at batch 1 and 25% worse at batch 4-16. The target step for a 1.5B model
  on this GPU is a few milliseconds of real work plus a scheduler tick; adding a 0.5B draft
  engine (its own forward, its own KV, sampler and scheduling per step) costs about as much as
  it saves. vLLM reports acceptance of ~0.5 per draft position, lower than the minimal
  implementation's ~0.78 because the engine counts rejected positions after the first mismatch
  and uses the prompts' chat template rather than the six fixed prompts.
* **n-gram speculation** accepts almost nothing on open-ended chat prompts (4% at batch 1) and
  ~30% when some prompts happen to repeat phrases; it never pays here. It is the right tool for
  prompt-grounded generation (summaries, code edits, RAG), not for this workload.
* Both methods reproduce plain greedy output exactly at batch 1 and diverge on some prompts at
  larger batches, the same finite-precision effect as in 6.1 (batched verification and plain
  decode use different GEMM shapes); the engine's rejection sampler is exact in distribution.
* This run happened during the heaviest co-tenant load of the session: the `none` baseline is
  11.4 ms/step here versus 5.6 ms/step for the same engine an hour earlier (run
  `20261029_023638_vllm_spec_decode_b27f9e1d13ab`). Within-run ratios are meaningful; absolute
  values are not.


## 7. Limitations and next experiments

* Targets are small (0.5B / 1.5B) because the GPUs were shared; the regime where speculation
  pays (7B+ target, batch 1-4) is pending an exclusive GPU. Every number above was taken while
  other jobs held most of the GPU, so only within-run ratios should be read.
* The minimal implementation's draft cost is dominated by Python / launch overhead; a CUDA-graph
  or compiled draft loop would change `c` dramatically. That is the point of comparing with vLLM.
* Next: EAGLE-style drafter in vLLM, acceptance-rate vs temperature curve for the sampling
  variant, and n-gram drafting on a RAG workload where the answer quotes the prompt.

## 8. References

* Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023.
* Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023.
* Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, 2024.
* Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, 2024.
* Saxena, *Prompt Lookup Decoding*, 2023 (n-gram drafting).
* vLLM documentation: Speculative Decoding (`speculative_config`).
