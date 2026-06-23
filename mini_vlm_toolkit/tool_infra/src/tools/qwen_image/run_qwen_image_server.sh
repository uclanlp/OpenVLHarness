#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

HOST="${QWEN_IMAGE_HOST:-127.0.0.1}"
PORT="${QWEN_IMAGE_PORT:-8088}"

exec uvicorn tools.qwen_image.server:app --host "$HOST" --port "$PORT"
