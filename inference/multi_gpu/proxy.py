"""Minimal round-robin HTTP proxy in front of N OpenAI-compatible replicas.

This is the "replica" side of the TP-vs-replica comparison: each backend is an independent
single-GPU server; the proxy only chooses a backend per request and streams the response
through unchanged. It is deliberately simple (no health-based routing, no least-loaded
policy) so that the comparison measures the engines, not the router.

    python -m inference.multi_gpu.proxy --port 8020 --backends http://127.0.0.1:8021,http://127.0.0.1:8022
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
from typing import Sequence

import aiohttp
from aiohttp import web


class RoundRobin:
    """Thread-unsafe round robin over backend URLs (one asyncio loop, so that's fine)."""

    def __init__(self, backends: Sequence[str]) -> None:
        if not backends:
            raise ValueError("need at least one backend")
        self.backends = [b.rstrip("/") for b in backends]
        self._cycle = itertools.cycle(range(len(self.backends)))
        self.counts = [0] * len(self.backends)

    def next(self) -> str:
        i = next(self._cycle)
        self.counts[i] += 1
        return self.backends[i]


def make_app(rr: RoundRobin) -> web.Application:
    app = web.Application()
    session: aiohttp.ClientSession | None = None

    async def on_startup(app: web.Application) -> None:
        nonlocal session
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=600))

    async def on_cleanup(app: web.Application) -> None:
        if session:
            await session.close()

    async def forward(request: web.Request) -> web.StreamResponse:
        assert session is not None
        backend = rr.next()
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}}
        async with session.post(backend + request.path_qs, data=body, headers=headers) as upstream:
            resp = web.StreamResponse(status=upstream.status, headers={"Content-Type": upstream.headers.get("Content-Type", "application/json")})
            await resp.prepare(request)
            async for chunk in upstream.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp

    async def health(request: web.Request) -> web.Response:
        assert session is not None
        statuses = {}
        for b in rr.backends:
            try:
                async with session.get(b + "/health", timeout=aiohttp.ClientTimeout(total=3)) as r:
                    statuses[b] = r.status
            except Exception as e:  # noqa: BLE001
                statuses[b] = f"error: {type(e).__name__}"
        ok = all(s == 200 for s in statuses.values())
        return web.json_response({"ok": ok, "backends": statuses, "counts": rr.counts}, status=200 if ok else 503)

    async def models(request: web.Request) -> web.StreamResponse:
        return await forward(request)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/completions", forward)
    app.router.add_post("/v1/chat/completions", forward)
    return app


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--backends", required=True, help="comma-separated backend base URLs")
    a = p.parse_args(argv)
    rr = RoundRobin(a.backends.split(","))
    print(f"[proxy] http://{a.host}:{a.port} -> {rr.backends}")
    web.run_app(make_app(rr), host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
    _ = asyncio
