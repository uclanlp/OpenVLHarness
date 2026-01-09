#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8101}"

exec uvicorn mini_vlm_toolkit.server.app:app --host "$HOST" --port "$PORT"
