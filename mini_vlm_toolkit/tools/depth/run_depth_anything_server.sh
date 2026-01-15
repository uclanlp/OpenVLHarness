#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR"

HOST="${DEPTH_ANYTHING_HOST:-127.0.0.1}"
PORT="${DEPTH_ANYTHING_PORT:-8082}"

exec uvicorn mini_vlm_toolkit.tools.depth.server:app --host "$HOST" --port "$PORT"
