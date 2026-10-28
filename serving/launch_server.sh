#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for the serving labs and wait until /health is up.
# Usage: serving/launch_server.sh [config.yaml] [extra python -m serving.server args...]
# Env:   CUDA_VISIBLE_DEVICES, PORT (default 8000), MODEL, GPU_UTIL
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/l03_server_baseline.yaml}"; shift || true
PY="${PY:-.venv/bin/python}"
ARGS=(--config "$CONFIG")
[[ -n "${PORT:-}" ]] && ARGS+=(--port "$PORT")
[[ -n "${MODEL:-}" ]] && ARGS+=(--model "$MODEL")
[[ -n "${GPU_UTIL:-}" ]] && ARGS+=(--gpu-memory-utilization "$GPU_UTIL")
exec "$PY" -m serving.server "${ARGS[@]}" "$@"
