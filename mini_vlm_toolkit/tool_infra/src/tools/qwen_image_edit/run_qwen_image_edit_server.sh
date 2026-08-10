#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

HOST="${QWEN_IMAGE_EDIT_HOST:-127.0.0.1}"
# 8090, not 8089: 8089 is Splunk's default mgmt port (collides on Splunk hosts).
PORT="${QWEN_IMAGE_EDIT_PORT:-8090}"

exec uvicorn tools.qwen_image_edit.server:app --host "$HOST" --port "$PORT"
