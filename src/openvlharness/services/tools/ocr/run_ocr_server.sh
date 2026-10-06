#!/usr/bin/env bash
# Launches vLLM as the PaddleOCR-VL OCR server.
# Exposes an OpenAI-compatible /v1/chat/completions endpoint that
# OCR_Tool (tool.py) calls directly.
#
# Optional environment variables:
#   OCR_MODEL_ID    (served identity, default: PaddlePaddle/PaddleOCR-VL-1.5)
#   OCR_MODEL_PATH  (optional local checkpoint directory; identity stays OCR_MODEL_ID)
#   OCR_HOST        (default: 127.0.0.1)
#   OCR_PORT        (default: 8085)

set -euo pipefail

MODEL_ID="${OCR_MODEL_ID:-PaddlePaddle/PaddleOCR-VL-1.5}"
MODEL_PATH="${OCR_MODEL_PATH:-$MODEL_ID}"
HOST="${OCR_HOST:-127.0.0.1}"
PORT="${OCR_PORT:-8085}"

exec python -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH" \
    --served-model-name "$MODEL_ID" \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.45 \
    --enforce-eager \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
