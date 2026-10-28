# L09. Application-to-Infrastructure Demo

**Question.** How do system-level choices (prompt layout, retrieval depth, reranking, tool calls,
verification) show up in an application's success rate and in its latency decomposition, and how do
those traces connect to the serving metrics of L02/L03?
**Status.** Implemented; RAG and agent evaluations *Measured (pilot, shared GPU / CPU)* on a
rag-mini-wikipedia subset with a 0.5B model. See section 6.
**Hardware needed.** CPU-only works (slow); one GPU with ~3 GB free for the 0.5B generator.

## 1. What the technique is

The lab builds the smallest complete application stack that exercises the serving layer in the
ways that matter for infrastructure:

```
question ─► retrieve (bge-small + FAISS) ─► rerank (bge-reranker-base, optional)
         ─► generate (Qwen2.5-0.5B-Instruct, HF or OpenAI-compatible endpoint)
agent:   plan/act loop over tools {search_passages, calculator, lookup_date} ─► verify claims
```

Every stage runs inside a span of `applications/trace.py` with a *kind* in
`{retrieval, rerank, model, tool, verify}`; the tracer reports seconds per kind, so a request's
latency is explained as *model compute* + *tool wait* + *retrieval* instead of one number.
OpenTelemetry plays this role in production; the local tracer keeps the semantics explicit and
exports Chrome trace JSON for Perfetto.

**Retrieval + reranking.** Dense retrieval embeds the query once (one small forward pass) and
does an inner-product search over 3 200 passages (microseconds for a flat index). A cross-encoder
reranker scores (query, passage) pairs jointly, which is more accurate than a bi-encoder but costs
one forward pass *per candidate*: reranking `k=10` candidates is ten small model calls. The
trade-off is precision of the top-`n` context versus latency and GPU time per request.

**Prompt layout and prefix caching.** The generator's prompt is `[static system prompt][retrieved
context][question]`. An engine with prefix caching (L02) reuses the KV cache of the longest shared
prefix, so the static part must come *first*; a per-request field before it (a user id, a date)
would make every prefix unique. Context tokens dominate prompt length, so retrieval depth (`top_n`)
sets the prefill cost per request.

**Tool-calling agent.** The model emits one JSON action per step (`{"action": tool, "args": …}` or
`{"action": "final", "answer": …}`); the loop executes the tool and appends the observation.
Native OpenAI-style tool calling is a flag for endpoints that support it. Each extra step is one
more model call *and* one more prompt that grows with the conversation, so cost is super-linear in
the number of steps.

**Claim verification.** The final answer is split into sentences; each is scored against the
evidence collected during the run (retrieved passages via the cross-encoder; tool observations by
checking that every number in the claim appears in an observation). Unsupported claims are flagged
or removed. Verification costs one reranker pass per claim × passage and can lower recall (true
claims without retrieved evidence are also removed) while raising precision.

## 2. When to use it

* **RAG** when answers depend on a corpus that is too large or too volatile for fine-tuning and
  when citations matter. Skip reranking when the retriever's top-`n` already contains the answer
  (measure `recall_in_context`); add it when candidates are noisy and latency allows.
* **A tool-calling agent** when the task needs computation or lookups the model cannot do
  reliably (arithmetic, dates, fresh facts) and when a small number of steps suffices; a fixed RAG
  pipeline is cheaper and more predictable when the workflow is known in advance.
* **Verification** when unsupported claims are expensive (user-facing facts, compliance) and the
  evidence is available in-context; not when answers are mostly subjective or when latency budgets
  are tight.
* **Traces of this shape** whenever an application is served by a shared engine: they are the
  only way to tell a slow model from a slow tool from a queue.

## 3. Trade-offs

| Advantages | Disadvantages |
|---|---|
| Retrieval grounds a small model in facts it does not store | Prompt length (and prefill cost) grows with `top_n`; recall@k bounds answer quality |
| Reranking improves the top-`n` context | One forward pass per candidate; latency scales with `k` |
| A static system-prompt prefix is cacheable across all requests | Any per-request content placed before it defeats the cache |
| Tools give exact arithmetic and fresh lookups | Each step is another model call with a longer prompt; small models often ignore the protocol |
| Verification removes unsupported claims | Costs reranker calls; can remove true-but-unretrieved claims; threshold needs calibration |
| Per-kind traces explain latency and join to serving metrics by run id | Another component to keep consistent with the serving layer's own metrics |

**Why agent success and serving throughput are different numbers.** Throughput counts tokens per
second a server can produce; success counts tasks a user got right. A configuration that doubles
throughput (bigger batches, shorter contexts) can lower success (less context, more truncation), and
a verification step that raises success lowers throughput (more calls per task). The report keeps
both, linked by the same run id, rather than substituting one for the other.

## 4. What is in this directory

| File | Purpose | Runs on |
|---|---|---|
| `trace.py` | Span tracer with kinds, per-kind breakdown, JSON + Chrome-trace export | CPU |
| `rag/index.py` | Build the FAISS index over rag-mini-wikipedia passages with bge-small | CPU/GPU |
| `rag/pipeline.py` | Retriever, Reranker, HF / OpenAI-compatible generators, `build_messages`, traced `RAGPipeline` | CPU/GPU |
| `rag/evaluate.py` | EM / F1 (SQuAD normalisation), recall@k, bootstrap CI, per-stage latency, with/without rerank | CPU/GPU |
| `agent/tools.py` | Safe AST calculator, date lookup, passage search tool, tool specs (JSON schema) | CPU |
| `agent/agent.py` | JSON-action (or native) tool loop, `Verifier`, claim support decision, traced `Agent` | CPU/GPU |
| `agent/evaluate.py` | Task success (QA + arithmetic) with verification off / flag / remove | CPU/GPU |
| `../tests/test_l09_*.py` | Tracer, scoring, parsing, calculator, verification, scripted agent loop (no downloads) | CPU |

## 5. How to run

All commands were executed from the repository root with `.venv/bin/python`.

```bash
python -m pytest tests/test_l09_trace.py tests/test_l09_rag.py tests/test_l09_agent.py -q

python applications/rag/index.py                       # 3 200 passages -> applications/rag/data/index
python applications/rag/evaluate.py --n 100 --rerank both --model Qwen/Qwen2.5-0.5B-Instruct
python applications/agent/evaluate.py --n-qa 40 --n-arith 10 --model Qwen/Qwen2.5-0.5B-Instruct

# Against an OpenAI-compatible server (vLLM/SGLang) so prefix-cache metrics are visible server-side:
python applications/rag/evaluate.py --generator endpoint --endpoint http://127.0.0.1:8000/v1 --model <served model>
python applications/agent/evaluate.py --generator endpoint --endpoint http://127.0.0.1:8000/v1 --model <served model> --native-tools
```

Each run writes `runs/L09_app_to_infra/<timestamp>_<name>_<cfghash>/` with `summary.json`,
per-question records, the last trace (`*.chrome.json` opens in Perfetto) and a latency figure.

## 6. Results

Single pilot runs; run directories are named in `reports/L09_app_to_infra/results/RESULTS.md`
(full tables, manifests, figures, last traces). Status: index **Measured**; RAG evaluation
**Measured (pilot, shared GPU)**; agent evaluation **Measured (pilot, shared GPU)**.

**6.1 Index.** 3 200 passages embedded with bge-small in 0.93 s on the GPU (3 426 passages/s);
model load 4.8 s dominates. Flat inner-product index, 384-d.

**6.2 RAG, 100 questions, Qwen2.5-0.5B-Instruct, k=10, top-3 context.**

| variant | EM (95% CI) | F1 | recall@10 | recall in top-3 | prompt tokens | wall / retrieval / rerank / model (ms) |
|---|---|---|---|---|---|---|
| no rerank | **0.57** [0.48, 0.66] | 0.63 | 0.51 | 0.44 | 409 | 45 / 6 / 0 / 38 |
| rerank (bge-reranker-base) | 0.48 [0.39, 0.58] | 0.54 | 0.51 | 0.44 | 424 | 55 / 5 / 16 / 34 |

Reranking **did not help here**: EM dropped by 9 points (13 answers flipped, net −9) at +10 ms per
question. Two caveats make this a pipeline finding rather than a verdict on rerankers: 53 of the
100 questions are yes/no (EM 0.79 without vs 0.70 with reranking), where substring recall cannot
tell whether the context supports the answer, and on the 47 open questions recall-in-context was
0.68 vs 0.70 (reranking slightly improved the context) while EM fell 0.32 → 0.23: the reranked
passage *order* changed which sentence the 0.5B model copied from. Latency is model-dominated
(38 ms of 45 ms); retrieval over the flat index is 5–6 ms including the query embedding; the
reranker's 10 pair-scorings cost 16 ms, i.e. ~40% of a question's budget.

**6.3 Agent, 40 QA + 10 arithmetic tasks, 3 steps max, JSON-action protocol.**

| verification | success (95% CI) | QA | arithmetic | model / tool calls | prompt tokens | unsupported claims | parse-fail | wall / model / tool / verify (s) |
|---|---|---|---|---|---|---|---|---|
| off | 0.24 [0.12, 0.36] | 0.10 | 0.80 | 2.96 / 1.96 | 1 381 | – | 0.26 | 1.34 / 1.33 / 0.005 / – |
| flag | 0.24 | 0.10 | 0.80 | 2.96 / 1.96 | 1 381 | 0.74 | 0.26 | 1.36 / 1.35 / 0.004 / 0.012 |
| remove | 0.24 | 0.10 | 0.80 | 2.96 / 1.96 | 1 381 | 0.74 | 0.26 | 1.36 / 1.35 / 0.004 / 0.012 |

* The tool loop works for arithmetic (0.80; the calculator answered every well-formed call in
  <0.1 ms) and fails for QA with this model: 13 of 40 QA runs ended in a truncated or non-JSON
  reply and the model often re-called a tool with its own observation instead of answering. The
  **same questions through the fixed RAG pipeline scored EM 0.57**; the agent spent 3.4× the prompt
  tokens (1 381 vs 409) and 3× the model calls for a 0.10 success rate. A 0.5B model should not
  drive an open-ended loop; the fixed pipeline is the right shape for it.
* Verification flagged 0.74 claims per task; in `remove` mode it changed 12 answers to `unknown`,
  **all 12 of which were already wrong**, so precision rose with no loss of correct answers and no
  change in success. Its cost was 12 ms per task (<1% of wall time) because the evidence was
  already in memory.
* Latency is 99% model compute (1.33 s of 1.34 s): tool wait is negligible for local tools. In a
  deployment with remote tools the `tool` span would carry the network wait and would be the number
  to compare against TTFT of the serving layer.

**6.4 From application traces to serving metrics.** Every trace carries the run id of the
evaluation; the generator's prompt starts with the same 60-token system prompt for every request,
so with prefix caching enabled on the serving side the server's cache-hit counter for that run
should be close to 1 for the system prompt and 0 for the context. The link was *not* measured in
this pilot because the shared GPU had no room for an engine; the `--generator endpoint` path and the
L03 server are the next experiment.

## 7. Limitations and next experiments

* **Small model, small subset.** Qwen2.5-0.5B-Instruct on 100 questions gives wide confidence
  intervals; the EM/F1 numbers characterise the pipeline, not the state of the art. The same scripts
  take a larger model through `--model` or `--endpoint`.
* **Local HF generation**, not an engine: per-request latency includes no batching or prefix
  caching. The infrastructure link (prefix-cache hit rate, TTFT under load) is measured by running
  the same evaluation against the L03 server with `--generator endpoint`; that run needs a GPU
  with room for the engine and was not possible on the shared GPU during this pilot.
* **Verification is heuristic** (cross-encoder relevance as a proxy for entailment, number
  matching for tool outputs). An NLI model or a judge call would be the next step, with its own
  cost line in the trace.
* **Agent protocol compliance** of small models is low; the evaluation reports the parse-failure
  rate so that protocol failures are not confused with reasoning failures.

## 8. References

* Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020.
* Nogueira & Cho, *Passage Re-ranking with BERT*, 2019 (cross-encoder reranking).
* Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023.
* Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*, SOSP 2023 (prefix sharing).
* rag-datasets/rag-mini-wikipedia, BAAI/bge-small-en-v1.5, BAAI/bge-reranker-base, Qwen/Qwen2.5-0.5B-Instruct on the Hugging Face Hub.
* OpenTelemetry tracing concepts: https://opentelemetry.io/docs/concepts/signals/traces/
