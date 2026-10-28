"""Helpers shared by the L08 vLLM/HF scripts: prompt loading, wikitext tokens, vLLM log capture."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any


def vllm_env() -> None:
    """Run the vLLM V1 engine core in-process.

    With multiprocessing on (the default) the engine lives in a child process, so the parent's
    `torch.cuda.memory_allocated()` and `logging` handlers see nothing. In-process mode lets a
    benchmark read weight/KV memory directly. Call before importing vllm.
    """
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class VllmLogCapture(logging.Handler):
    """Collects the vLLM log lines that carry memory numbers not exposed through the API.

    Captures: "Model loading took X GiB", "GPU KV cache size: N tokens",
    "Maximum concurrency for N tokens per request: Kx", "Graph capturing finished in X secs".
    """

    PATTERNS = {
        "model_weights_gib": re.compile(r"Model loading took ([\d.]+) GiB"),
        "kv_cache_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
        "max_concurrency": re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"),
        "cuda_graph_capture_s": re.compile(r"Graph capturing finished in ([\d.]+) secs"),
        "cuda_graph_memory_gib": re.compile(r"took ([\d.]+) GiB"),
    }

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.values: dict[str, Any] = {}
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        self.lines.append(msg)
        for key, pat in self.PATTERNS.items():
            m = pat.search(msg)
            if not m:
                continue
            if key == "max_concurrency":
                self.values["max_concurrency_tokens_per_request"] = int(m.group(1).replace(",", ""))
                self.values["max_concurrency_x"] = float(m.group(2))
            elif key == "cuda_graph_memory_gib" and "Graph capturing" not in msg:
                continue
            else:
                self.values[key] = float(m.group(1).replace(",", ""))

    def install(self) -> VllmLogCapture:
        logging.getLogger("vllm").addHandler(self)
        return self

    def remove(self) -> None:
        logging.getLogger("vllm").removeHandler(self)


def load_prompts(n: int, min_chars: int = 40, max_chars: int = 1200, seed: int = 0) -> list[str]:
    """First user turns of `HuggingFaceH4/ultrachat_200k` (test_sft), filtered by length."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:2000]")
    ds = ds.shuffle(seed=seed)
    out: list[str] = []
    for row in ds:
        msg = row["messages"][0]
        if msg["role"] != "user":
            continue
        text = msg["content"].strip()
        if min_chars <= len(text) <= max_chars:
            out.append(text)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)} prompts of length {min_chars}-{max_chars}")
    return out


def chat_prompts(tokenizer, prompts: list[str]) -> list[str]:
    """Apply the model's chat template so instruct models behave as intended."""
    return [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]


def wikitext_token_ids(tokenizer, max_tokens: int | None = None):
    """wikitext-2-raw-v1 test set concatenated with blank lines and tokenised once."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if max_tokens:
        ids = ids[:max_tokens]
    return ids


def resolve_checkpoint(model_id: str, load: str) -> str:
    """Map (base model, load mode) to the checkpoint to load.

    Pre-quantised AWQ / GPTQ checkpoints are separate repos on the Hub; everything else loads the
    bf16 base checkpoint and quantises (or not) at load time.
    """
    if load == "awq":
        return model_id if "AWQ" in model_id else f"{model_id}-AWQ"
    if load == "gptq":
        return model_id if "GPTQ" in model_id else f"{model_id}-GPTQ-Int4"
    if load == "gptq_int8":
        return model_id if "GPTQ" in model_id else f"{model_id}-GPTQ-Int8"
    return model_id


def free_gpu_memory_gib(device: int = 0) -> float:
    import torch

    free, _ = torch.cuda.mem_get_info(device)
    return free / 2**30


def wait_for_free_memory(min_gib: float, timeout_s: float = 3600, poll_s: float = 10, device: int = 0) -> float:
    """Block until the GPU has at least `min_gib` free (this machine's GPUs are shared).

    Returns the free memory when the wait ended. Raises TimeoutError after `timeout_s`.
    """
    import time

    import torch

    if min_gib <= 0 or not torch.cuda.is_available():
        return float("inf")
    t0 = time.time()
    while True:
        free = free_gpu_memory_gib(device)
        if free >= min_gib:
            return free
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"waited {timeout_s}s for {min_gib} GiB free GPU memory, have {free:.2f} GiB")
        print(f"[wait] {free:.2f} GiB free < {min_gib} GiB needed; retrying in {poll_s}s", flush=True)
        time.sleep(poll_s)


def build_llm(kwargs: dict, attempts: int = 3, retry_wait_s: float = 15):
    """`vllm.LLM(**kwargs)` with retries for the shared-GPU profiling race.

    vLLM sizes the KV cache by profiling free memory before/after a dummy forward and asserts
    that no *other* process changed free memory meanwhile ("Error in memory profiling"). On a
    shared GPU that race is routine. Passing `kv_cache_memory_bytes` skips the profiling-based
    sizing entirely, which is what the L08 scripts do by default; the retry covers the rest.
    """
    import time

    from vllm import LLM

    last: Exception | None = None
    for i in range(attempts):
        try:
            return LLM(**kwargs)
        except AssertionError as e:
            if "memory profiling" not in str(e):
                raise
            last = e
            print(f"[build_llm] memory-profiling race (attempt {i + 1}/{attempts}); retrying in {retry_wait_s}s", flush=True)
            time.sleep(retry_wait_s)
    raise RuntimeError(f"vLLM init failed after {attempts} attempts: {last}")


def run_in_subprocess(script: str, args: list[str], out: Path, timeout_s: float = 1800, attempts: int = 3, retry_wait_s: float = 60) -> dict:
    """Run `script --one ... --out out` in a child interpreter and load its JSON result.

    Why: vLLM's in-process engine does not release all GPU memory on `del llm`, so building several
    engines in one process accumulates memory (observed: 11 GiB held by a process whose single
    engine needs ~5 GiB). One engine per process is the only reliable isolation, and it also
    turns a crash into a recorded error instead of a lost run.

    On a shared GPU a child can still OOM when a co-tenant grabs memory between the free-memory
    check and the allocation; such failures are retried up to `attempts` times.
    """
    import json
    import subprocess
    import sys
    import time

    out = Path(out)
    cmd = [sys.executable, script, *args, "--out", str(out)]
    res: dict = {}
    for i in range(attempts):
        if out.exists():
            out.unlink()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if out.exists():
            res = json.loads(out.read_text())
        else:
            tail = (proc.stderr.strip().splitlines() or proc.stdout.strip().splitlines() or ["no output"])[-1][:400]
            res = {"status": "failed", "error": f"exit {proc.returncode}: {tail}"}
        err = str(res.get("error", ""))
        transient = res.get("status") != "ok" and any(k in err for k in ("out of memory", "Free memory on device", "memory profiling"))
        if not transient or i == attempts - 1:
            break
        print(f"[run_in_subprocess] transient GPU memory failure (attempt {i + 1}/{attempts}): {err[:120]}; retrying in {retry_wait_s}s", flush=True)
        time.sleep(retry_wait_s)
    res["attempts"] = i + 1
    return res
