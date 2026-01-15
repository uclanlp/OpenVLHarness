#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"

ENV_FILE="$ROOT_DIR/.env"
PRE_HOST="${HOST-}"
PRE_PORT="${PORT-}"
PRE_TOOL_MODULES="${TOOL_MODULES-}"
PRE_HEAVY_TOOL_MODULES="${HEAVY_TOOL_MODULES-}"
PRE_SAM3_SERVER_URL="${SAM3_SERVER_URL-}"
PRE_GROUNDING_DINO_SERVER_URL="${GROUNDING_DINO_SERVER_URL-}"
PRE_OCR_SERVER_URL="${OCR_SERVER_URL-}"
PRE_DEPTH_ANYTHING_SERVER_URL="${DEPTH_ANYTHING_SERVER_URL-}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  set +u
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set -u
  set +a
fi

if [[ -n "$PRE_HOST" ]]; then
  HOST="$PRE_HOST"
fi
if [[ -n "$PRE_PORT" ]]; then
  PORT="$PRE_PORT"
fi
if [[ -n "$PRE_TOOL_MODULES" ]]; then
  TOOL_MODULES="$PRE_TOOL_MODULES"
fi
if [[ -n "$PRE_HEAVY_TOOL_MODULES" ]]; then
  HEAVY_TOOL_MODULES="$PRE_HEAVY_TOOL_MODULES"
fi
if [[ -n "$PRE_SAM3_SERVER_URL" ]]; then
  SAM3_SERVER_URL="$PRE_SAM3_SERVER_URL"
  export SAM3_SERVER_URL
fi
if [[ -n "$PRE_GROUNDING_DINO_SERVER_URL" ]]; then
  GROUNDING_DINO_SERVER_URL="$PRE_GROUNDING_DINO_SERVER_URL"
  export GROUNDING_DINO_SERVER_URL
fi
if [[ -n "$PRE_OCR_SERVER_URL" ]]; then
  OCR_SERVER_URL="$PRE_OCR_SERVER_URL"
  export OCR_SERVER_URL
fi
if [[ -n "$PRE_DEPTH_ANYTHING_SERVER_URL" ]]; then
  DEPTH_ANYTHING_SERVER_URL="$PRE_DEPTH_ANYTHING_SERVER_URL"
  export DEPTH_ANYTHING_SERVER_URL
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8101}"

if [[ -z "${TOOL_MODULES:-}" ]]; then
  echo "TOOL_MODULES is required. Set it in $ENV_FILE or export it before running." >&2
  exit 1
fi

if [[ -n "${HEAVY_TOOL_MODULES:-}" ]]; then
  TOOL_MODULES="${TOOL_MODULES},${HEAVY_TOOL_MODULES}"
fi
export TOOL_MODULES

exec env TOOL_MODULES="$TOOL_MODULES" uvicorn mini_vlm_toolkit.server.app:app --host "$HOST" --port "$PORT"
