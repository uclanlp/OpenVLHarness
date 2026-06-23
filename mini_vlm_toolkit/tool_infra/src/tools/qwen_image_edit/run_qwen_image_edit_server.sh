#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

HOST="${QWEN_IMAGE_EDIT_HOST:-127.0.0.1}"
PORT="${QWEN_IMAGE_EDIT_PORT:-8089}"

exec uvicorn tools.qwen_image_edit.server:app --host "$HOST" --port "$PORT"
