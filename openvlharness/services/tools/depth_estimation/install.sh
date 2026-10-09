#!/usr/bin/env bash
# Separate DA3 environment; depth and pose share the same model/server.
set -euo pipefail
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  printf '%s\n' 'Activate a dedicated venv or conda environment before running this installer.' >&2
  exit 2
fi
SOURCE_ROOT="${OPENVLHARNESS_SOURCE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
if [[ ! -f "$SOURCE_ROOT/pyproject.toml" ]]; then
  printf '%s\n' 'Set OPENVLHARNESS_SOURCE to the OpenVLHarness source checkout.' >&2
  exit 2
fi
python -m pip install 'torch==2.10.0' 'torchvision==0.25.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'torch==2.10.0' 'torchvision==0.25.0' \
  'depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4' \
  'numpy<2' matplotlib python-multipart fastapi uvicorn
python -m pip install "$SOURCE_ROOT"
