#!/usr/bin/env bash
# Install into an already activated, dedicated Python environment. No model downloads.
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
  'sam3 @ git+https://github.com/facebookresearch/sam3.git@2345a4ad109ac29c569da749c91d84f10dc08c40' \
  'setuptools<81' einops psutil fastapi uvicorn python-multipart pycocotools \
  'opencv-python-headless<4.12' matplotlib scipy
python -m pip install "$SOURCE_ROOT"
