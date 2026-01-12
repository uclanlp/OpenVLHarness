#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8101}"

DEFAULT_TOOL_MODULES=(
  "mini_vlm_toolkit.tools.python_executor.tool"
  "mini_vlm_toolkit.tools.gemini_google_search.tool"
  "mini_vlm_toolkit.tools.serper_google_search.tool"
  "mini_vlm_toolkit.tools.single_url_qa.tool"
  "mini_vlm_toolkit.tools.python_coder.tool"
  "mini_vlm_toolkit.tools.crop.tool"
  "mini_vlm_toolkit.tools.overlay_images.tool"
  "mini_vlm_toolkit.tools.drawline.tool"
  "mini_vlm_toolkit.tools.draw_horizontal_line_by_y.tool"
  "mini_vlm_toolkit.tools.draw_vertical_line_by_x.tool"
  "mini_vlm_toolkit.tools.point.tool"
  "mini_vlm_toolkit.tools.zoom_in_subfigure.tool"
  "mini_vlm_toolkit.tools.zoom_in_image_by_bbox.tool"
)

HEAVY_TOOL_MODULES=(
    #"mini_vlm_toolkit.tools.sam3_segment.tool"
    "mini_vlm_toolkit.tools.grounding_dino.tool"
)

if [[ -z "${TOOL_MODULES:-}" ]]; then
  TOOL_MODULES="$(IFS=,; printf "%s" "${DEFAULT_TOOL_MODULES[*]}")"
fi
export TOOL_MODULES

# To include heavy tools, append them manually, for example:
TOOL_MODULES="${TOOL_MODULES},$(IFS=,; printf "%s" "${HEAVY_TOOL_MODULES[*]}")"

exec env TOOL_MODULES="$TOOL_MODULES" uvicorn mini_vlm_toolkit.server.app:app --host "$HOST" --port "$PORT"
