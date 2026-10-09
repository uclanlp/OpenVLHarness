#!/usr/bin/env bash
set -euo pipefail
# Optional extension; select a GPU with CUDA_VISIBLE_DEVICES.
HOST="${GROUNDING_DINO_HOST:-127.0.0.1}"
PORT="${GROUNDING_DINO_PORT:-8084}"

exec python -m uvicorn openvlharness.services.tools.grounding_dino.server:app --host "$HOST" --port "$PORT" "$@"
