# Mini VLM Tool Kit

Minimal research toolkit with a tool server and client.

## Tutorial

See `TUTORIAL.md` for endpoint layout and client usage examples (including LLM-assisted tool calls).

## Quick Start

Install dependencies:
```bash
pip install -r requirements-client.txt
```

Optional SAM3 server dependencies:
```bash
pip install -r requirements-server.txt
```

Start the server:
```bash
export PYTHONPATH="$(pwd)"
./scripts/run_server.sh
```

Start the SAM3 model server (required for the SAM3 tool):
```bash
export PYTHONPATH="$(pwd)"
./scripts/run_sam3_server.sh
```

Start the server with a specific tool list:
```bash
export PYTHONPATH="$(pwd)"
export TOOL_MODULES="mini_vlm_toolkit.tools.python_executor.tool,mini_vlm_toolkit.tools.sam3_segment.tool"
./scripts/run_server.sh
```

List tools:
```bash
curl http://127.0.0.1:8101/tools
```
By default the server loads the tool modules listed in `mini_vlm_toolkit/tool_registry/loader.py`.

## CLI (no Python required)

Local execution (no server):
```bash
python -m mini_vlm_toolkit.cli --tool Python_Executor_Tool --code "print(2+2)\nvalue=2+2"
```

Server execution:
```bash
python -m mini_vlm_toolkit.cli --server http://127.0.0.1:8101 --tool Python_Executor_Tool --code "print(5*3)\nvalue=15"
```

## Tests

Each tool has a standalone test script. The Python executor test includes:
1) direct tool invocation
2) client invocation via server
3) LLM-generated tool call via client
   - The LLM-assisted step builds a prompt and parses tool args in the test script.
This direct + client + LLM-assisted pattern should be the default for new tool tests.

Run:
```bash
export PYTHONPATH="$(pwd)"
python tests/test_python_executor.py
```
Direct execution helper test:
```bash
export PYTHONPATH="$(pwd)"
python tests/test_execute_direct.py
```

Additional vision-tool tests (each downloads sample images into `tests/data` on first run and saves outputs to `tests/output`):
```bash
export PYTHONPATH="$(pwd)"
python tests/test_crop_tool.py
python tests/test_drawline_tool.py
python tests/test_draw_horizontal_line_by_y_tool.py
python tests/test_draw_vertical_line_by_x_tool.py
python tests/test_grounding_dino_tool.py
python tests/test_overlay_images_tool.py
python tests/test_point_tool.py
python tests/test_zoom_in_subfigure_tool.py
python tests/test_zoom_in_image_by_bbox_tool.py
```

LLM test environment:
- `LLM_MODEL` (optional, default Qwen/Qwen3-8B)
- `LLM_BASE_URL` (optional, default http://localhost:8004/v1)
- `LLM_API_KEY` (optional, default "local")

Server-based tests expect the tool server to be running or `TOOL_SERVER_URL` to point to a running instance. 
Some tools require API keys in the files listed in Notes below.

Example for vLLM Qwen3-8B at port 8004:
```bash
export LLM_MODEL=Qwen/Qwen3-8B
export LLM_BASE_URL=http://localhost:8004/v1
```

## TODO
- Incorporate tools from the Visual Sketchpad paper (ported in-process; no external servers).
- Incorporate tools from OpenThinkIMG (https://arxiv.org/pdf/2505.08617).
  - GroundingDINO (HF server wrapper, base model) [completed]
  - OCR
  - crop (normalized coords primary; pixel coords optional) [completed]
  - point [completed]
  - SegmentRegionAroundPoint (implement via SAM3, point-guided)
  - ZoomInSubfigure (subplot selection by description) [completed]
  - drawline (region + corner indices) [completed]
  - DrawHorizontalLineByY [completed]
  - DrawVerticalLineByX [completed]
- Incorporate tools from VisualSketchpad (https://arxiv.org/abs/2406.09403).
  - segment_and_mark (full-image segmentation with numbered masks)
  - zoom_in_image_by_bbox (bbox-based zoom) [completed]
  - depth (depth estimation)
  - overlay_images (alpha blend an overlay into a base image) [completed]
  - sliding_window_detection (small-object helper that runs detection on crops)

## Notes
- The executor runs code with `exec()` and no sandboxing.
- Run commands from the repo root or set `PYTHONPATH="$(pwd)"` so imports resolve.
- `sam3_segment` expects its extra dependencies (`requirements-server.txt`) and a working GPU setup.
- GroundingDINO runs as a separate model server; set `GROUNDING_DINO_SERVER_URL` and launch `./scripts/run_grounding_dino_server.sh` (requires `torch`, `torchvision`, and `transformers`).
- API key files are read from:
  - `mini_vlm_toolkit/tools/gemini_google_search/google_api_key`
  - `mini_vlm_toolkit/tools/serper_google_search/serper_api_key`
  - `mini_vlm_toolkit/tools/single_url_qa/openai_api_key`
- Reminder: run `python tests/test_gemini_google_search.py`, `python tests/test_serper_google_search.py`, and `python tests/test_single_url_qa.py`.
