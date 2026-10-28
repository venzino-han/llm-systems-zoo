"""L07.8 — Ray Serve: replicas, request routing and Serve-level batching.

Serve owns the HTTP front door, the per-replica queue (`max_ongoing_requests`), routing and
autoscaling. It does *not* own the model's batching: a vLLM engine batches internally; an HF model
only batches if the deployment does so with `@serve.batch`. This module exposes a small LLM
deployment with knobs for both so the two layers can be compared.

Modes:
  --fake       no model: sleeps `--fake-ms` per request (CPU tests, scheduling-only experiments)
  default      HF model (tiny/small) per replica, optional `@serve.batch` dynamic batching
  --proxy URL  forwards to an OpenAI-compatible server (e.g. vLLM); Serve only routes

Run:  python distributed/ray_serve/deployment.py --fake --replicas 2 --port 8001
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ray  # noqa: E402
from ray import serve  # noqa: E402
from starlette.requests import Request  # noqa: E402

from distributed.ray_core._common import cuda_available  # noqa: E402
from llm_systems_zoo.registry import MODELS  # noqa: E402


class LLMDeploymentImpl:
    def __init__(self, model_id: str | None, max_new_tokens: int, fake_ms: float, proxy: str | None, batch_wait_ms: float, max_batch: int) -> None:
        self.max_new_tokens = max_new_tokens
        self.fake_ms = fake_ms
        self.proxy = proxy
        self.model = None
        self.n_batches = 0
        self.n_items = 0
        if proxy:
            import httpx

            self.client = httpx.AsyncClient(base_url=proxy, timeout=120)
            self.proxy_model = model_id
        elif model_id and not fake_ms:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = "cuda" if cuda_available() else "cpu"
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.tok.padding_side = "left"
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32).to(self.device).eval()
        # Dynamic batching is configured per instance so CLI flags control it.
        self._batched = serve.batch(max_batch_size=max_batch, batch_wait_timeout_s=batch_wait_ms / 1e3)(self._generate_batch)

    async def _generate_batch(self, prompts: list[str]) -> list[dict[str, Any]]:
        self.n_batches += 1
        self.n_items += len(prompts)
        if self.fake_ms:
            await asyncio.sleep(self.fake_ms / 1e3)
            return [{"text": "ok", "output_tokens": 1, "batch": len(prompts)} for _ in prompts]
        import torch

        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        loop = asyncio.get_running_loop()

        def _gen():
            with torch.no_grad():
                return self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)

        out = await loop.run_in_executor(None, _gen)
        texts = self.tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return [{"text": t, "output_tokens": int((out[i, enc["input_ids"].shape[1] :] != self.tok.pad_token_id).sum()), "batch": len(prompts)} for i, t in enumerate(texts)]

    async def _proxy(self, prompt: str) -> dict[str, Any]:
        r = await self.client.post("/v1/completions", json={"model": self.proxy_model, "prompt": prompt, "max_tokens": self.max_new_tokens, "temperature": 0})
        j = r.json()
        return {"text": j["choices"][0]["text"], "output_tokens": j.get("usage", {}).get("completion_tokens", 0), "batch": None}

    async def __call__(self, request: Request) -> dict[str, Any]:
        body = await request.json()
        t0 = time.perf_counter()
        prompt = body.get("prompt", "")
        out = await (self._proxy(prompt) if self.proxy else self._batched(prompt))
        out["replica_s"] = time.perf_counter() - t0
        out["n_batches"] = self.n_batches
        out["n_items"] = self.n_items
        return out


def build_app(model_id: str | None, replicas: int, max_ongoing: int, max_new_tokens: int, fake_ms: float, proxy: str | None, batch_wait_ms: float, max_batch: int, gpu_frac: float, autoscale: dict[str, Any] | None = None):
    opts: dict[str, Any] = {"max_ongoing_requests": max_ongoing, "ray_actor_options": {"num_cpus": 1}}
    if autoscale:
        opts["autoscaling_config"] = autoscale
    else:
        opts["num_replicas"] = replicas
    if cuda_available() and not fake_ms and not proxy:
        opts["ray_actor_options"]["num_gpus"] = gpu_frac
    Deployment = serve.deployment(**opts)(LLMDeploymentImpl)
    return Deployment.bind(model_id, max_new_tokens, fake_ms, proxy, batch_wait_ms, max_batch)


def start(app, port: int, route: str = "/generate") -> str:
    serve.start(detached=False, http_options={"host": "127.0.0.1", "port": port})
    serve.run(app, name="llm", route_prefix=route)
    return f"http://127.0.0.1:{port}{route}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODELS["small"].hf_id)
    ap.add_argument("--replicas", type=int, default=1)
    ap.add_argument("--max-ongoing", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--fake-ms", type=float, default=50)
    ap.add_argument("--proxy", default=None, help="OpenAI-compatible base URL to forward to")
    ap.add_argument("--batch-wait-ms", type=float, default=0, help="Serve-level dynamic batching window (0 = off)")
    ap.add_argument("--max-batch", type=int, default=1)
    ap.add_argument("--gpu-frac", type=float, default=0.2)
    ap.add_argument("--autoscale", action="store_true")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--serve-seconds", type=float, default=300)
    args = ap.parse_args()
    autoscale = {"min_replicas": 1, "max_replicas": max(2, args.replicas), "target_ongoing_requests": 4, "upscale_delay_s": 2, "downscale_delay_s": 10} if args.autoscale else None
    ray.init(num_cpus=8, num_gpus=1 if cuda_available() else 0, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    app = build_app(None if args.fake else args.model, args.replicas, args.max_ongoing, args.max_new_tokens, args.fake_ms if args.fake else 0, args.proxy, args.batch_wait_ms, args.max_batch, args.gpu_frac, autoscale)
    url = start(app, args.port)
    print(f"serving at {url} (replicas={args.replicas}, autoscale={bool(autoscale)}); Ctrl-C to stop", flush=True)
    try:
        time.sleep(args.serve_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
