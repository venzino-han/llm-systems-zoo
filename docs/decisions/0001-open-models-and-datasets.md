# ADR-0001: Use ungated, permissively licensed open models and datasets

**Status.** Accepted (2026-10-29)

## Context
The repository must be reproducible by anyone, without a Hugging Face token, gated-model approval or
proprietary API keys. It must also be legally safe to publish results and small artefacts.

## Decision
All labs default to the models and datasets in `llm_systems_zoo/registry.py`:

* Qwen2.5-Instruct family (0.5B / 1.5B / 7B, Apache-2.0), including its pre-quantised AWQ and GPTQ checkpoints,
  which gives one tokenizer family across the size range (needed for speculative decoding).
* SmolLM2-135M-Instruct (Apache-2.0) for CPU smoke tests.
* BGE small / reranker base (MIT / Apache-2.0) for retrieval.
* wikitext-2, smoltalk, ultrafeedback_binarized, GSM8K, ultrachat_200k, rag-mini-wikipedia.

Run manifests pin the model revision used for every published number.

## Consequences
* No Llama-family models by default (gated). Scripts accept any HF id, so a user with access can swap them in.
* Results from 0.5B–7B models do not generalise to 70B+ behaviour; every report states its model size.
