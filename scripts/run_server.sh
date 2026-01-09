#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"

exec uvicorn mini_vlm_tool_kit.server.app:app --host "$HOST" --port "$PORT"
