#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8083}"

exec uvicorn sam3_server:app --host "$HOST" --port "$PORT" --app-dir "$SCRIPT_DIR"
