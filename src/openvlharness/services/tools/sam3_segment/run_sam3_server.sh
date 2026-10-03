#!/usr/bin/env bash
set -euo pipefail
# Select a GPU with CUDA_VISIBLE_DEVICES in the raw SAM3 environment.
HOST="${SAM3_HOST:-127.0.0.1}"
PORT="${SAM3_PORT:-8086}"

exec python -m uvicorn openvlharness.services.tools.sam3_segment.server:app --host "$HOST" --port "$PORT" "$@"
