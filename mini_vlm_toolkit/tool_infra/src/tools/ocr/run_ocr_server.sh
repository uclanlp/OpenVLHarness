#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

HOST="${OCR_HOST:-127.0.0.1}"
PORT="${OCR_PORT:-8085}"

exec uvicorn tools.ocr.server:app --host "$HOST" --port "$PORT"
