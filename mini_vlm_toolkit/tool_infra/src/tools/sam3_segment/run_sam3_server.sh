#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

HOST="${SAM3_HOST:-127.0.0.1}"
PORT="${SAM3_PORT:-8086}"

exec uvicorn tools.sam3_segment.server:app --host "$HOST" --port "$PORT"
