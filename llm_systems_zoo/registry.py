"""Open models and datasets used across the labs.

Everything here is ungated on the Hugging Face Hub and redistributable under a permissive
licence, so any lab can be reproduced without a token. Pin `revision` in run manifests when
you publish a number; the identifiers below are the defaults a script uses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    params_b: float
    license: str
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    # CPU smoke tests / unit tests: fits in a few hundred MB, loads in seconds.
    "tiny": ModelSpec("HuggingFaceTB/SmolLM2-135M-Instruct", 0.135, "Apache-2.0", "CPU smoke tests"),
    # Default for single-GPU experiments: fast enough to sweep many conditions.
    "small": ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", 0.49, "Apache-2.0", "default experiment model"),
    "medium": ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", 1.54, "Apache-2.0", ""),
    # Large enough that quantisation / TP / speculative decoding effects are visible.
    "large": ModelSpec("Qwen/Qwen2.5-7B-Instruct", 7.6, "Apache-2.0", "target model for spec decode, TP"),
    "large_awq": ModelSpec("Qwen/Qwen2.5-7B-Instruct-AWQ", 7.6, "Apache-2.0", "W4 AWQ, pre-quantised"),
    "large_gptq_int4": ModelSpec("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", 7.6, "Apache-2.0", "W4 GPTQ, pre-quantised"),
    "small_awq": ModelSpec("Qwen/Qwen2.5-0.5B-Instruct-AWQ", 0.49, "Apache-2.0", ""),
    "small_gptq_int4": ModelSpec("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", 0.49, "Apache-2.0", ""),
    # Same tokenizer family as "large": usable as a draft model for speculative decoding.
    "draft": ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", 0.49, "Apache-2.0", "draft for Qwen2.5-7B target"),
    # Embedding / reranking models for the application lab.
    "embed": ModelSpec("BAAI/bge-small-en-v1.5", 0.033, "MIT", "dense retriever"),
    "rerank": ModelSpec("BAAI/bge-reranker-base", 0.278, "Apache-2.0", "cross-encoder reranker"),
}


@dataclass(frozen=True)
class DatasetSpec:
    hf_id: str
    config: str | None
    split: str
    license: str
    notes: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "lm_eval": DatasetSpec("wikitext", "wikitext-2-raw-v1", "test", "CC-BY-SA-3.0", "perplexity"),
    "sft": DatasetSpec("HuggingFaceTB/smoltalk", "everyday-conversations", "train", "Apache-2.0", "chat SFT"),
    "dpo": DatasetSpec("trl-lib/ultrafeedback_binarized", None, "train", "MIT", "preference pairs"),
    "math": DatasetSpec("openai/gsm8k", "main", "test", "MIT", "GRPO reward + accuracy eval"),
    "chat_prompts": DatasetSpec("HuggingFaceH4/ultrachat_200k", None, "test_sft", "MIT", "serving workload prompts"),
    "rag_corpus": DatasetSpec("rag-datasets/rag-mini-wikipedia", "text-corpus", "passages", "CC-BY-SA-3.0", "RAG passages"),
    "rag_qa": DatasetSpec("rag-datasets/rag-mini-wikipedia", "question-answer", "test", "CC-BY-SA-3.0", "RAG questions"),
}


def model_id(key: str) -> str:
    return MODELS[key].hf_id
