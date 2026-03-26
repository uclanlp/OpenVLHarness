#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

HOST="${DEPTH_ESTIMATION_HOST:-127.0.0.1}"
PORT="${DEPTH_ESTIMATION_PORT:-8087}"

exec uvicorn tools.depth_estimation.server:app --host "$HOST" --port "$PORT"
