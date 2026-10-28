# L03 report: serving SLO lab

Status: **harness implemented and tested on CPU; pilot measured on a shared GPU at 10 % memory.**
Full measurement on an exclusive GPU is pending.

## Question
What is the maximum request load the server sustains while meeting a latency SLO (TTFT ≤ 0.5 s,
TPOT ≤ 0.05 s), where does the saturation knee sit, and what do batch width, chunked prefill and
client-side admission control change?

## Environment
* Host `kirc`, GPU 0: NVIDIA RTX A6000 48 GiB (sm_86), shared with other processes during the pilot
  (another tenant held ~27 GiB; our server was limited to 10 % of the device).
* vLLM 0.23.0, torch 2.11.0+cu130, transformers 5.7.0, Python 3.10.
* Model: `Qwen/Qwen2.5-0.5B-Instruct`, bf16, `--enforce-eager`, `--max-model-len 2048`,
  `max_num_seqs 256`, `max_num_batched_tokens 8192`, chunked prefill on, prefix caching on
  (`configs/l03_server_pilot_lowmem.yaml`).
* Workload `short_qa`: prompt length lognormal(median 128, σ 0.4, 16–512), output lognormal(median 128,
  σ 0.3, 16–256), `ignore_eos` so output length equals `max_tokens`; exact prompt lengths via the
  model tokenizer; seed 0. Warm-up requests excluded.

## Hypothesis
1. Closed-loop throughput grows almost linearly with concurrency while decode is dominated by
   per-step fixed cost, and TPOT grows slowly; the knee appears when the step becomes compute-bound
   or the KV cache fills.
2. Open-loop TTFT stays flat below the knee and then explodes (queue growth, Little's law), while
   TPOT stays bounded because the scheduler caps the batch.
3. At 2× the saturation rate, a client-side in-flight cap restores SLO goodput at the price of an
   explicit rejection rate.

## Baseline
The pilot server configuration above, driven closed-loop at concurrency 1.

## Change
Load only (concurrency / arrival rate) for the saturation curves; admission limit for the overload
experiment. Server configuration is fixed within each sweep.

## Method
`serving/run_sweep.py concurrency-sweep` (closed loop) and `rate-sweep` (open loop, Poisson), each
point a harness run with per-request records; analysis with `serving/slo_analysis.py`. Sections 6 of
`serving/README.md` lists the exact commands.

## Results
Full tables and figures: `serving/README.md` §6 and `results/*/saturation.md|png|sweep.json`.

| Experiment | Key numbers (pilot, shared GPU, 10 % memory, eager) |
|---|---|
| Closed-loop sweep (c = 1…256) | linear to c=128: 75.5 req/s, 9914 out tok/s, TPOT p50 10 ms; c=256 flat at 75.1 req/s, TTFT p50 381 ms, attainment 0.79 |
| Open-loop sweep (10…160 req/s) | knee between 40 and 80 req/s; 80 req/s: 58.8 achieved, TTFT p99 497 ms, attainment 0.99; 120 req/s: TTFT p50 3.8 s, attainment 0.13; 160 req/s: throughput falls to 57.4 req/s |
| Overload at 140 req/s | no admission: goodput 7.6 req/s, TTFT p50 7.7 s; cap 128: goodput 34.6 req/s with 1126 rejections; cap 64: TTFT p99 117 ms, goodput 28.5 req/s, 1253 rejections |
| `max_num_seqs` 256 vs 8 at 16 req/s | TPOT p50 11 vs 7 ms; goodput 15.0 vs 0.37 req/s; narrow scheduler saturates at ≈8 req/s and queues |

Hypotheses 1–3 are supported by the pilot. Hypothesis 1's knee is a step-time (compute) limit, not a
KV-cache limit: the server log reports 275,696 KV tokens of capacity and no preemptions.

## Bottleneck analysis
* **Below the knee** TTFT is flat (~25–45 ms ≈ one prefill plus scheduler latency) and TPOT is set by
  the decode step. Throughput is linear in concurrency because the 0.5B model's step is dominated by
  fixed per-step cost in eager mode (kernel launches, sampling, scheduler), not by weight reads.
* **At the knee** (c≈128, ≈70 req/s) the step time grows with the batch (TPOT p50 10→17 ms from 128 to
  256 in flight), so extra concurrency buys nothing and only lengthens the queue.
* **Beyond the knee in open loop** the damage is entirely queueing delay: TTFT grows by seconds while
  TPOT rises by milliseconds, because the scheduler bounds the batch and the waiting queue absorbs the
  excess. At 160 req/s served throughput even drops, consistent with overheads from a very large
  waiting set and longer contexts per step.
* **Admission control** converts that queueing delay into explicit rejections; goodput improves 3.8–4.5×
  at 2× overload. The cap must be sized from capacity × SLO budget (Little's law).
* **Narrow batches** cut TPOT but cap capacity; past that capacity the result is worse than the wide
  scheduler on every metric except TPOT.
* **Confound:** the GPU was shared; TPOT at a fixed concurrency varied by ~2× between sweeps
  (8 ms vs 15 ms at 64 in flight), so only within-sweep comparisons are reliable.

## Limits
* Shared GPU, 10 % memory, eager mode: absolute throughput is not representative of the hardware.
* p99 values from points with fewer than 100 requests are flagged in the tables.
* Single run per point; no spread reported yet.
* Client-side measurements only (no server-side step timing or GPU counters joined yet).

## Reproduction commands
See `serving/README.md` §5. Raw runs: `runs/L03_serving_slo/`; curated copies in `results/`.

## Next experiments
1. Exclusive GPU, `gpu_memory_utilization 0.9`, CUDA graphs on: repeat both sweeps ×3 and report the spread.
2. `mixed_burst` with chunked prefill on vs off (TPOT p99 of short requests during long prefills).
3. `long_prefix_shared` with prefix caching on vs off (TTFT and prefill tokens/s).
4. `max_num_seqs` 8 vs 256 at fixed open-loop load (TPOT vs queueing).
5. A 7B model, where decode is clearly memory-bound and the knee is sharper.
