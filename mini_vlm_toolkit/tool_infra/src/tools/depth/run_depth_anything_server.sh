#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

HOST="${DEPTH_ANYTHING_HOST:-127.0.0.1}"
PORT="${DEPTH_ANYTHING_PORT:-8082}"

exec uvicorn tools.depth.server:app --host "$HOST" --port "$PORT"
