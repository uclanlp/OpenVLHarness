#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8101}"
# all modules:
# "mini_vlm_toolkit.tools.python_executor.tool,mini_vlm_toolkit.tools.gemini_google_search.tool,mini_vlm_toolkit.tools.serper_google_search.tool,mini_vlm_toolkit.tools.single_url_qa.tool,mini_vlm_toolkit.tools.python_coder.tool,mini_vlm_toolkit.tools.sam3_segment.tool,mini_vlm_toolkit.tools.crop.tool,mini_vlm_toolkit.tools.overlay_images.tool,mini_vlm_toolkit.tools.drawline.tool,mini_vlm_toolkit.tools.draw_horizontal_line_by_y.tool,mini_vlm_toolkit.tools.draw_vertical_line_by_x.tool,mini_vlm_toolkit.tools.point.tool,mini_vlm_toolkit.tools.zoom_in_subfigure.tool,mini_vlm_toolkit.tools.zoom_in_image_by_bbox.tool"
TOOL_MODULES="mini_vlm_toolkit.tools.python_executor.tool,mini_vlm_toolkit.tools.gemini_google_search.tool,mini_vlm_toolkit.tools.serper_google_search.tool,mini_vlm_toolkit.tools.single_url_qa.tool,mini_vlm_toolkit.tools.python_coder.tool,mini_vlm_toolkit.tools.crop.tool,mini_vlm_toolkit.tools.overlay_images.tool,mini_vlm_toolkit.tools.drawline.tool,mini_vlm_toolkit.tools.draw_horizontal_line_by_y.tool,mini_vlm_toolkit.tools.draw_vertical_line_by_x.tool,mini_vlm_toolkit.tools.point.tool,mini_vlm_toolkit.tools.zoom_in_subfigure.tool,mini_vlm_toolkit.tools.zoom_in_image_by_bbox.tool"
export TOOL_MODULES

exec env TOOL_MODULES="$TOOL_MODULES" uvicorn mini_vlm_toolkit.server.app:app --host "$HOST" --port "$PORT"
