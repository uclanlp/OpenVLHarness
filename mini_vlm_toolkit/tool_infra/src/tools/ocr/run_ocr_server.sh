#!/bin/bash
# Launches vLLM as the PaddleOCR-VL OCR server.
# Exposes an OpenAI-compatible /v1/chat/completions endpoint that
# OCR_Tool (tool.py) calls directly.
#
# Optional environment variables:
#   OCR_MODEL_ID    (default: PaddlePaddle/PaddleOCR-VL-1.5)
#   OCR_HOST        (default: 127.0.0.1)
#   OCR_PORT        (default: 8085)

set -euo pipefail

MODEL_ID="${OCR_MODEL_ID:-PaddlePaddle/PaddleOCR-VL-1.5}"
HOST="${OCR_HOST:-127.0.0.1}"
PORT="${OCR_PORT:-8085}"

exec vllm serve "$MODEL_ID" \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
