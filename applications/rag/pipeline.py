"""L09.2 — RAG pipeline: retrieve -> (rerank) -> generate, every stage traced.

Prompt layout matters for serving: the *static* system prompt comes first, then the retrieved
context, then the question. With prefix caching (L02) the system-prompt prefix is shared by every
request; putting anything request-specific before it would defeat the cache.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from applications.rag.index import DEFAULT_INDEX_DIR, load_index  # noqa: E402
from applications.trace import Tracer  # noqa: E402
from llm_systems_zoo.registry import MODELS  # noqa: E402

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer using only the provided context. "
    "Reply with the shortest possible answer: a name, a date, a number, yes/no, or a short phrase. "
    "If the context does not contain the answer, reply exactly: unknown."
)


def build_messages(question: str, passages: list[str], system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    ctx = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return [
        {"role": "system", "content": system_prompt},  # static prefix: cacheable across requests
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"},
    ]


class Retriever:
    def __init__(self, index_dir: Path = DEFAULT_INDEX_DIR, embed_model: str | None = None, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.index, self.passages, self.meta = load_index(Path(index_dir))
        self.model = SentenceTransformer(embed_model or self.meta["embed_model"], device=device)

    def retrieve(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        import numpy as np

        q = np.asarray(self.model.encode([query], normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        scores, ids = self.index.search(q, k)
        return [{"rank": r, "id": self.passages[i]["id"], "passage": self.passages[i]["passage"], "score": float(s)} for r, (i, s) in enumerate(zip(ids[0], scores[0])) if i >= 0]


class Reranker:
    def __init__(self, model_id: str = MODELS["rerank"].hf_id, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_id, device=device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts], show_progress_bar=False)]

    def rerank(self, query: str, cands: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        scores = self.score(query, [c["passage"] for c in cands])
        order = sorted(range(len(cands)), key=lambda i: -scores[i])
        return [{**cands[i], "rerank_score": scores[i]} for i in order[:top_n]]


class HFGenerator:
    """Local transformers generation (greedy). Fine for pilots; use an engine for throughput."""

    def __init__(self, model_id: str = MODELS["small"].hf_id, device: str | None = None, max_new_tokens: int = 32) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int | None = None, **_: Any) -> dict[str, Any]:
        import torch

        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id)
        gen = out[0, ids["input_ids"].shape[1] :]
        return {"text": self.tok.decode(gen, skip_special_tokens=True).strip(), "prompt_tokens": int(ids["input_ids"].shape[1]), "output_tokens": int(gen.shape[0])}


class OpenAIGenerator:
    """Any OpenAI-compatible endpoint (vLLM, SGLang, Ray Serve proxy)."""

    def __init__(self, endpoint: str, model_id: str = MODELS["small"].hf_id, max_new_tokens: int = 32, api_key: str = "EMPTY") -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=endpoint, api_key=api_key)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int | None = None, tools: list[dict] | None = None) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        if tools:
            kw["tools"] = tools
        r = self.client.chat.completions.create(model=self.model_id, messages=messages, max_tokens=max_new_tokens or self.max_new_tokens, temperature=0, **kw)
        msg = r.choices[0].message
        out = {"text": (msg.content or "").strip(), "prompt_tokens": r.usage.prompt_tokens if r.usage else 0, "output_tokens": r.usage.completion_tokens if r.usage else 0}
        if getattr(msg, "tool_calls", None):
            out["tool_calls"] = [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in msg.tool_calls]
        return out


def make_generator(kind: str, model_id: str, endpoint: str | None = None, max_new_tokens: int = 32, device: str | None = None):
    if kind == "endpoint":
        if not endpoint:
            raise ValueError("--endpoint is required for the endpoint generator")
        return OpenAIGenerator(endpoint, model_id, max_new_tokens)
    return HFGenerator(model_id, device, max_new_tokens)


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Any, reranker: Reranker | None = None, tracer: Tracer | None = None, k: int = 10, top_n: int = 3) -> None:
        self.retriever, self.generator, self.reranker = retriever, generator, reranker
        self.tracer = tracer or Tracer()
        self.k, self.top_n = k, top_n

    def answer(self, question: str) -> dict[str, Any]:
        tr = self.tracer
        with tr.span("rag.request", "other", question=question):
            with tr.span("retrieve", "retrieval", k=self.k) as s:
                cands = self.retriever.retrieve(question, self.k)
                s.set(n=len(cands))
            if self.reranker is not None:
                with tr.span("rerank", "rerank", top_n=self.top_n):
                    ctx = self.reranker.rerank(question, cands, self.top_n)
            else:
                ctx = cands[: self.top_n]
            messages = build_messages(question, [c["passage"] for c in ctx])
            with tr.span("generate", "model") as s:
                t0 = time.perf_counter()
                g = self.generator.generate(messages)
                s.set(prompt_tokens=g["prompt_tokens"], output_tokens=g["output_tokens"], gen_s=time.perf_counter() - t0)
        return {"question": question, "answer": g["text"], "context": ctx, "candidates": cands, "prompt_tokens": g["prompt_tokens"], "output_tokens": g["output_tokens"]}
