# ADR-0002: vLLM is the primary inference engine; others are adapters added later

**Status.** Accepted (2026-10-29)

## Context
The plan calls for one main engine until the shared harness (workloads, load generator, metrics) is
stable, then comparisons. Candidates: vLLM, SGLang, TensorRT-LLM, HF transformers.

## Decision
* **vLLM** (OpenAI-compatible server and offline `LLM` API) is the engine for the KV-cache, serving,
  multi-GPU and most L08 experiments.
* **HF transformers** is used where the experiment needs to see inside the forward pass (memory
  anatomy, profiler traces, minimal speculative decoding, torch.compile).
* SGLang / TensorRT-LLM comparisons are deferred until the harness and the first reports are published.

## Consequences
* Engine-specific flags live in the lab scripts; the load generator and metrics are engine-agnostic
  (any OpenAI-compatible endpoint).
* Educational minimal implementations (paged allocator, speculative decoder, TP linear layers, Triton
  kernels) are always labelled as such and never compared one-to-one with the engine's numbers.
