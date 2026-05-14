#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

# HOST can be set by conda's build toolchain to a compiler triplet (e.g.
# x86_64-conda-linux-gnu) which is not a valid bind address.  Clear it so the
# .env file's value is used unless the caller explicitly wants an override.
unset HOST

# Accept env file name as argument, default to .env
ENV_FILE_NAME="${1:-.env}"
ENV_FILE="$ROOT_DIR/$ENV_FILE_NAME"
PRE_HOST="${HOST-}"
PRE_PORT="${PORT-}"
PRE_TOOL_MODULES="${TOOL_MODULES-}"
PRE_LIGHT_TOOL_MODULES="${LIGHT_TOOL_MODULES-}"
PRE_HEAVY_TOOL_MODULES="${HEAVY_TOOL_MODULES-}"
PRE_SAM3_SERVER_URL="${SAM3_SERVER_URL-}"
PRE_GROUNDING_DINO_SERVER_URL="${GROUNDING_DINO_SERVER_URL-}"
PRE_OCR_SERVER_URL="${OCR_SERVER_URL-}"
PRE_DEPTH_ESTIMATION_SERVER_URL="${DEPTH_ESTIMATION_SERVER_URL-}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing .env file at $ENV_FILE. Create it from .env.example." >&2
  exit 1
fi

ENV_FILE_REALPATH="$(realpath "$ENV_FILE")"

set -a
set +u
# shellcheck disable=SC1090
source "$ENV_FILE"
set -u
set +a

if [[ -n "$PRE_HOST" ]]; then
  HOST="$PRE_HOST"
fi
if [[ -n "$PRE_PORT" ]]; then
  PORT="$PRE_PORT"
fi
if [[ -n "$PRE_LIGHT_TOOL_MODULES" ]]; then
  LIGHT_TOOL_MODULES="$PRE_LIGHT_TOOL_MODULES"
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
if [[ -n "$PRE_DEPTH_ESTIMATION_SERVER_URL" ]]; then
  DEPTH_ESTIMATION_SERVER_URL="$PRE_DEPTH_ESTIMATION_SERVER_URL"
  export DEPTH_ESTIMATION_SERVER_URL
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8101}"
export HOST PORT

if [[ -n "$PRE_TOOL_MODULES" || -n "${TOOL_MODULES:-}" ]]; then
  echo "TOOL_MODULES is not read by scripts/run_server.sh. Use LIGHT_TOOL_MODULES/HEAVY_TOOL_MODULES instead." >&2
  exit 1
fi

if [[ -z "${LIGHT_TOOL_MODULES:-}" && -z "${HEAVY_TOOL_MODULES:-}" ]]; then
  echo "Set LIGHT_TOOL_MODULES or HEAVY_TOOL_MODULES in $ENV_FILE or export them before running." >&2
  exit 1
fi

if [[ "${LIGHT_TOOL_MODULES:-}" == "auto" ]]; then
  if [[ -n "${HEAVY_TOOL_MODULES:-}" ]]; then
    echo "HEAVY_TOOL_MODULES is ignored when LIGHT_TOOL_MODULES=auto." >&2
  fi
  MODE="auto"
  TOOL_MODULES="auto"
else
  MODE="explicit"
  TOOL_MODULES="${LIGHT_TOOL_MODULES:-}"
  if [[ -n "${HEAVY_TOOL_MODULES:-}" ]]; then
    if [[ -n "${TOOL_MODULES}" ]]; then
      TOOL_MODULES="${TOOL_MODULES},${HEAVY_TOOL_MODULES}"
    else
      TOOL_MODULES="${HEAVY_TOOL_MODULES}"
    fi
  fi
fi
echo "Loaded env: ${ENV_FILE_REALPATH} (mode=${MODE})" >&2
export TOOL_MODULES

exec env TOOL_MODULES="$TOOL_MODULES" uvicorn server.app:app --host "$HOST" --port "$PORT"
