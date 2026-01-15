#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR"

HOST="${GROUNDING_DINO_HOST:-127.0.0.1}"
PORT="${GROUNDING_DINO_PORT:-8084}"

exec uvicorn mini_vlm_toolkit.tools.grounding_dino.server:app --host "$HOST" --port "$PORT"
