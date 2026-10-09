#!/usr/bin/env bash
set -euo pipefail
# One process provides both /v1/depth and /v1/pose.
HOST="${DEPTH_ESTIMATION_HOST:-127.0.0.1}"
PORT="${DEPTH_ESTIMATION_PORT:-8087}"

exec python -m uvicorn openvlharness.services.tools.depth_estimation.server:app --host "$HOST" --port "$PORT" "$@"
