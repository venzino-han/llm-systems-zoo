"""Start / stop a vLLM OpenAI-compatible server from Python with the scheduler knobs that
matter for serving experiments exposed as a config.

    python -m serving.server --config configs/l03_server_baseline.yaml --port 8000

The scheduler flags and what they control:
    --max-num-seqs            max requests in one decode step (batch width)
    --max-num-batched-tokens  token budget per scheduler step (prefill chunk size + decode tokens)
    --enable-chunked-prefill  split long prefills into chunks so decode steps are not stalled
    --enable-prefix-caching   reuse KV blocks for identical prompt prefixes
    --gpu-memory-utilization  fraction of the GPU reserved for weights + activations + KV cache
    --kv-cache-memory-bytes   explicit KV cache budget (overrides the utilisation-derived one)
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import httpx
import yaml


@dataclass
class ServerConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    port: int = 8000
    host: str = "127.0.0.1"
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.3
    kv_cache_memory_bytes: int | None = None
    max_model_len: int | None = 4096
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    enable_chunked_prefill: bool | None = None
    enable_prefix_caching: bool | None = None
    enforce_eager: bool = False
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    seed: int = 0
    served_model_name: str | None = None
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)  # e.g. CUDA_VISIBLE_DEVICES
    label: str = "baseline"

    @classmethod
    def load(cls, path: str | Path, **overrides: Any) -> ServerConfig:
        d = yaml.safe_load(Path(path).read_text()) or {}
        d.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def command(self, python: str | None = None) -> list[str]:
        cmd = [python or sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", self.model,
               "--host", self.host, "--port", str(self.port), "--dtype", self.dtype, "--seed", str(self.seed),
               "--gpu-memory-utilization", str(self.gpu_memory_utilization),
               "--tensor-parallel-size", str(self.tensor_parallel_size), "--disable-log-stats"]
        if self.data_parallel_size > 1:
            cmd += ["--data-parallel-size", str(self.data_parallel_size)]
        if self.kv_cache_memory_bytes:
            cmd += ["--kv-cache-memory-bytes", str(self.kv_cache_memory_bytes)]
        if self.max_model_len:
            cmd += ["--max-model-len", str(self.max_model_len)]
        if self.max_num_seqs:
            cmd += ["--max-num-seqs", str(self.max_num_seqs)]
        if self.max_num_batched_tokens:
            cmd += ["--max-num-batched-tokens", str(self.max_num_batched_tokens)]
        if self.enable_chunked_prefill is not None:
            cmd.append("--enable-chunked-prefill" if self.enable_chunked_prefill else "--no-enable-chunked-prefill")
        if self.enable_prefix_caching is not None:
            cmd.append("--enable-prefix-caching" if self.enable_prefix_caching else "--no-enable-prefix-caching")
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        if self.served_model_name:
            cmd += ["--served-model-name", self.served_model_name]
        cmd += list(self.extra_args)
        return cmd

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def model_name(self) -> str:
        return self.served_model_name or self.model


def wait_for_health(url: str, timeout_s: float = 600, proc: subprocess.Popen | None = None, interval: float = 2.0) -> float:
    """Block until GET {url}/health returns 200. Returns the seconds waited."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            if httpx.get(f"{url}/health", timeout=2.0).status_code == 200:
                return time.perf_counter() - t0
        except httpx.HTTPError:
            pass
        time.sleep(interval)
    raise TimeoutError(f"server at {url} not healthy after {timeout_s}s")


class VLLMServer:
    """Context manager around a vLLM API server subprocess (own process group, log file)."""

    def __init__(self, config: ServerConfig, log_dir: str | Path = "runs/_server_logs", startup_timeout_s: float = 900, startup_retries: int = 1):
        self.config = config
        self.log_dir = Path(log_dir)
        self.startup_timeout_s = startup_timeout_s
        self.startup_retries = startup_retries  # vLLM's memory profiling can fail transiently on a shared GPU
        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.startup_s: float | None = None

    def start(self) -> VLLMServer:
        last: Exception | None = None
        for attempt in range(self.startup_retries + 1):
            try:
                return self._start_once()
            except RuntimeError as e:
                last = e
                if attempt < self.startup_retries:
                    print(f"[server] start failed (attempt {attempt + 1}), retrying in 10s: {str(e).splitlines()[0]}", flush=True)
                    time.sleep(10)
        assert last is not None
        raise last

    def _start_once(self) -> VLLMServer:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"vllm_{self.config.label}_{self.config.port}_{int(time.time())}.log"
        env = {**os.environ, **self.config.env}
        cmd = self.config.command()
        print(f"[server] starting: {shlex.join(cmd)}\n[server] log: {self.log_path}", flush=True)
        with self.log_path.open("w") as log:
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            self.startup_s = wait_for_health(self.config.url, self.startup_timeout_s, self.proc)
        except Exception:
            self.stop()
            tail = self.log_path.read_text()[-3000:] if self.log_path.exists() else ""
            raise RuntimeError(f"vLLM server failed to start; log tail:\n{tail}") from None
        print(f"[server] ready in {self.startup_s:.1f}s at {self.config.url} (pid {self.proc.pid})", flush=True)
        return self

    def stop(self, grace_s: float = 15) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        pgid = os.getpgid(self.proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            self.proc.wait(grace_s)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            self.proc.wait(10)
        print(f"[server] stopped pid {self.proc.pid}", flush=True)

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def __enter__(self) -> VLLMServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="YAML ServerConfig (configs/l03_server_*.yaml)")
    p.add_argument("--model", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--enforce-eager", action="store_true", default=None)
    p.add_argument("--print-command", action="store_true")
    a = p.parse_args(argv)
    overrides = {k: v for k, v in vars(a).items() if k not in {"config", "print_command"}}
    cfg = ServerConfig.load(a.config, **overrides) if a.config else ServerConfig(**{k: v for k, v in overrides.items() if v is not None})
    if a.print_command:
        print(shlex.join(cfg.command()))
        return
    # SIGTERM (e.g. `kill <launcher pid>`) must stop the vLLM process group too.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with VLLMServer(cfg) as server:
        print("[server] Ctrl-C or SIGTERM to stop")
        try:
            while server.proc and server.proc.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
