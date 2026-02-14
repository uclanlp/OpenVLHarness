#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"
APP_DIR="$ROOT_DIR/src/tools/sam3_segment"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8083}"

exec uvicorn sam3_server:app --host "$HOST" --port "$PORT" --app-dir "$APP_DIR"
