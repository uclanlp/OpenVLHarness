#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

HOST="${GROUNDING_DINO_HOST:-127.0.0.1}"
PORT="${GROUNDING_DINO_PORT:-8084}"

exec uvicorn tools.grounding_dino.server:app --host "$HOST" --port "$PORT"
