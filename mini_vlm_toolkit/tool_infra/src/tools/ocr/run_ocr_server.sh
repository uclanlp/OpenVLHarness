#!/bin/bash
# Launches the PaddleOCR-VL server.
#
# Optional environment variables:
#   OCR_MODEL_ID           (default: PaddlePaddle/PaddleOCR-VL-1.5)
#   OCR_DEVICE             (auto | cuda | cpu, default: auto)
#   OCR_TORCH_DTYPE        (default: bfloat16; e.g. float16, float32)
#   OCR_MAX_NEW_TOKENS     (default: 2048)
#   OCR_REPETITION_PENALTY (default: 1.0; PaddleOCR-VL recommends 1.05+)
#   OCR_HOST               (default: 127.0.0.1)
#   OCR_PORT               (default: 8085)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

HOST="${OCR_HOST:-127.0.0.1}"
PORT="${OCR_PORT:-8085}"

exec uvicorn tools.ocr.server:app --host "$HOST" --port "$PORT"
