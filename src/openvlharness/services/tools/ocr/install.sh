#!/usr/bin/env bash
# PaddleOCR-VL is served by vLLM; this path does not use the PaddlePaddle runtime.
set -euo pipefail
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  printf '%s\n' 'Activate a dedicated venv or conda environment before running this installer.' >&2
  exit 2
fi
python -m pip install 'vllm==0.18.1' 'transformers==4.57.3'
