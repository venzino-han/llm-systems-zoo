"""A tiny OpenAI-compatible *fake* streaming server used for CPU tests and smoke runs.

It models a serving engine with a fixed capacity: at most `max_concurrency` requests are
"decoding" at once (the rest queue), each request pays `ttft_s` before its first token and
`tpot_s` per subsequent token. No model is involved. This is enough to exercise the load
generator's bookkeeping and to show queueing / saturation behaviour on a laptop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from typing import Any

from aiohttp import web


def _token_count(text: str) -> int:
    return max(1, len(text.split()))


class FakeEngine:
    def __init__(self, max_concurrency: int = 8, ttft_s: float = 0.02, tpot_s: float = 0.005, queue_limit: int | None = None):
        self.max_concurrency = max_concurrency
        self.ttft_s = ttft_s
        self.tpot_s = tpot_s
        self.queue_limit = queue_limit
        self.sem: asyncio.Semaphore | None = None
        self.waiting = 0
        self.served = 0

    def _semaphore(self) -> asyncio.Semaphore:
        if self.sem is None:
            self.sem = asyncio.Semaphore(self.max_concurrency)
        return self.sem

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body: dict[str, Any] = await request.json()
        is_chat = request.path.endswith("/chat/completions")
        if is_chat:
            prompt = " ".join(m.get("content", "") for m in body.get("messages", []))
        else:
            prompt = body.get("prompt", "")
        prompt_tokens = _token_count(prompt)
        max_tokens = int(body.get("max_tokens", 16))
        stream = bool(body.get("stream", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        if self.queue_limit is not None and self.waiting >= self.queue_limit:
            return web.json_response({"error": {"message": "queue full"}}, status=503)

        self.waiting += 1
        sem = self._semaphore()
        async with sem:
            self.waiting -= 1
            await asyncio.sleep(self.ttft_s)
            rid = f"fake-{self.served}"
            self.served += 1
            if not stream:
                await asyncio.sleep(self.tpot_s * max(0, max_tokens - 1))
                text = " ".join(["tok"] * max_tokens)
                payload = self._final(rid, is_chat, text, prompt_tokens, max_tokens)
                return web.json_response(payload)

            resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for i in range(max_tokens):
                if i > 0:
                    await asyncio.sleep(self.tpot_s)
                chunk = self._chunk(rid, is_chat, "tok ", finish=(i == max_tokens - 1))
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            if include_usage:
                usage = {"id": rid, "object": "chat.completion.chunk" if is_chat else "text_completion", "choices": [],
                         "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens, "total_tokens": prompt_tokens + max_tokens}}
                await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp

    @staticmethod
    def _chunk(rid: str, is_chat: bool, text: str, finish: bool) -> dict[str, Any]:
        fr = "stop" if finish else None
        if is_chat:
            return {"id": rid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": fr}]}
        return {"id": rid, "object": "text_completion", "choices": [{"index": 0, "text": text, "finish_reason": fr}]}

    @staticmethod
    def _final(rid: str, is_chat: bool, text: str, pt: int, ct: int) -> dict[str, Any]:
        usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
        if is_chat:
            return {"id": rid, "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage}
        return {"id": rid, "object": "text_completion", "choices": [{"index": 0, "text": text, "finish_reason": "stop"}], "usage": usage}


def make_app(engine: FakeEngine) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/completions", engine.handle)
    app.router.add_post("/v1/chat/completions", engine.handle)
    app.router.add_get("/health", lambda r: web.Response(text="ok"))
    app.router.add_get("/v1/models", lambda r: web.json_response({"data": [{"id": "fake-model"}]}))
    return app


class FakeServerThread:
    """Run the fake server on a background thread with its own event loop (for tests)."""

    def __init__(self, engine: FakeEngine | None = None, host: str = "127.0.0.1", port: int = 0):
        self.engine = engine or FakeEngine()
        self.host, self.port = host, port
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ready = threading.Event()
        self._runner: web.AppRunner | None = None

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start())
        self._ready.set()
        self._loop.run_forever()

    async def _start(self) -> None:
        self._runner = web.AppRunner(make_app(self.engine))
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    def start(self) -> FakeServerThread:
        self._thread.start()
        self._ready.wait(10)
        return self

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        async def _cleanup():
            if self._runner:
                await self._runner.cleanup()

        fut = asyncio.run_coroutine_threadsafe(_cleanup(), self._loop)
        try:
            fut.result(5)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(5)

    def __enter__(self) -> FakeServerThread:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main() -> None:
    p = argparse.ArgumentParser(description="Fake OpenAI-compatible streaming server")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--ttft", type=float, default=0.02)
    p.add_argument("--tpot", type=float, default=0.005)
    p.add_argument("--queue-limit", type=int, default=None)
    a = p.parse_args()
    engine = FakeEngine(a.max_concurrency, a.ttft, a.tpot, a.queue_limit)
    print(f"fake server on http://127.0.0.1:{a.port} (cap={a.max_concurrency}, ttft={a.ttft}s, tpot={a.tpot}s)")
    web.run_app(make_app(engine), host="127.0.0.1", port=a.port, print=None)


if __name__ == "__main__":
    main()
    _ = time
