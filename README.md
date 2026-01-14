# Mini VLM Tool Kit

Minimal research toolkit with a tool server and client.

## Tutorial

See `TUTORIAL.md` for endpoint layout and client usage examples (including LLM-assisted tool calls).

## Implemented Tools

| Tool | Description |
| --- | --- |
| `Crop_Tool` | Crop an image using normalized or pixel coordinates. |
| `Depth_Tool` | Estimate depth from an image using a Depth-Anything server. |
| `DrawHorizontalLineByY_Tool` | Draw a horizontal line at the provided y coordinate. |
| `DrawLine_Tool` | Draw a dashed line between two corners of a bounding box. |
| `DrawVerticalLineByX_Tool` | Draw a vertical line at the provided x coordinate. |
| `Gemini_Google_Search_Tool` | Gemini-backed Google search tool with optional citations. |
| `GroundingDINO_Tool` | Detect objects in an image using a GroundingDINO server and return bounding boxes. |
| `OCR_Tool` | Extract text strings from an image using an OCR server. |
| `OverlayImages_Tool` | Overlay an image onto another with alpha blending. |
| `Point_Tool` | Extract point coordinates from text or provided points and return normalized coordinates. |
| `Python_Code_Generator_Tool` | Generate and execute simple Python code snippets for basic arithmetic calculations and math-related problems. |
| `Python_Executor_Tool` | Execute a provided Python code snippet and return its output and variables. |
| `SAM3_Segment_Tool` | Call an external SAM3 segmentation server with a prompt and image. |
| `SAM3_VisualizeAllMasks_Tool` | Overlay all SAM3 masks on an image (grounding-agent or SAM3 server format). |
| `SAM3_VisualizeSingleMask_Tool` | Overlay a single SAM3 mask and return a zoomed-in view. |
| `SegmentAndMark_Tool` | Segment the full image and annotate each mask with a numeric label. |
| `SegmentRegionAroundPoint_Tool` | Segment a region around one or more points using SAM3. |
| `Serper_Google_Search_Tool` | Serper-backed Google search tool that returns URLs. |
| `Single_URL_QA_Tool` | Answer a query using content retrieved from a single URL. |
| `SlidingWindowDetection_Tool` | Run detection on overlapping crops to find small objects. |
| `ZoomInImageByBBox_Tool` | Crop an image around a normalized bbox with optional padding. |
| `ZoomInSubfigure_Tool` | Select and return subplots that match a textual description. |

## Quick Start

Install dependencies:
```bash
pip install -r requirements-client.txt
```

Optional heavy tool dependencies (SAM3, GroundingDINO, OCR, Depth):
```bash
pip install -r requirements-server.txt
```
SAM3 visualization tools are lightweight (Pillow/numpy). If you pass grounding-agent RLE masks,
they require `pycocotools` from the server requirements.

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

Start the OCR model server (required for the OCR tool):
```bash
export PYTHONPATH="$(pwd)"
./scripts/run_ocr_server.sh
```

Start the GroundingDINO model server (required for the GroundingDINO tool):
```bash
export PYTHONPATH="$(pwd)"
./scripts/run_grounding_dino_server.sh
```

Start the Depth-Anything model server (required for the Depth tool):
```bash
export PYTHONPATH="$(pwd)"
./scripts/run_depth_anything_server.sh
```

Start the server with a specific tool list:
```bash
export PYTHONPATH="$(pwd)"
export TOOL_MODULES="mini_vlm_toolkit.tools.python_executor.tool,mini_vlm_toolkit.tools.sam3_segment.tool,mini_vlm_toolkit.tools.sam3_visualize.tool"
./scripts/run_server.sh
```
Note: `mini_vlm_toolkit.tools.sam3_visualize.tool` defines two tools:
`SAM3_VisualizeAllMasks_Tool` and `SAM3_VisualizeSingleMask_Tool`.

List tools:
```bash
curl http://127.0.0.1:8101/tools
```
By default `scripts/run_server.sh` sets `TOOL_MODULES` to a curated list (see the script). If you run `uvicorn` directly, the fallback tool list is in `mini_vlm_toolkit/tool_registry/loader.py`.

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

```bash
export PYTHONPATH="$(pwd)"
python tests/test_python_executor.py
```

- Run any tool test with `python tests/test_<tool>.py`.
- Server-based tests need the tool server running or `TOOL_SERVER_URL`.
- LLM-assisted tests read `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY` (defaults in code).
- Vision tests download sample images into `tests/data` and write outputs to `tests/output`.
- Web/URL QA tests require API keys (see Notes).

Example LLM env:
```bash
export LLM_MODEL=Qwen/Qwen3-8B
export LLM_BASE_URL=http://localhost:8004/v1
```

## References
- `reference/` holds read-only snapshots of upstream repos; do not edit them for project changes.
- `reference/AgentFlow`
- `reference/OpenThinkIMG`
- `reference/VisualSketchpad`
- `reference/tool-vision`
- `reference/GroundingDINO`
- `reference/dpr-reproduction`

## Notes
- The executor runs code with `exec()` and no sandboxing.
- Run commands from the repo root or set `PYTHONPATH="$(pwd)"` so imports resolve.
- If you are using Di's shared server (always running on port 8101), use the Serper search, cloud search, and OpenAI-related tools with caution because they use Di's personal API key.
- `sam3_segment` expects its extra dependencies (`requirements-server.txt`) and a working GPU setup.
- GroundingDINO runs as a separate model server; set `GROUNDING_DINO_SERVER_URL` and launch `./scripts/run_grounding_dino_server.sh` (requires `torch`, `torchvision`, and `transformers`).
- OCR runs as a separate EasyOCR server; set `OCR_SERVER_URL` and launch `./scripts/run_ocr_server.sh` (requires `easyocr`, downloads weights on first run, defaults to CPU).
- Depth runs as a separate Depth-Anything server; set `DEPTH_ANYTHING_SERVER_URL` and launch `./scripts/run_depth_anything_server.sh` (requires `torch` and `transformers`, optional `matplotlib` for Inferno colormap).
- API key files are read from:
  - `mini_vlm_toolkit/tools/gemini_google_search/google_api_key`
  - `mini_vlm_toolkit/tools/serper_google_search/serper_api_key`
  - `mini_vlm_toolkit/tools/single_url_qa/openai_api_key`
- Reminder: run `python tests/test_gemini_google_search.py`, `python tests/test_serper_google_search.py`, and `python tests/test_single_url_qa.py`.
- SAM3 visualize tests: `python tests/test_sam3_visualize.py` (requires SAM3 + tool server) or `python tests/test_sam3_visualize_synthetic.py` (tool server only).

## TODO
- SAM3: return an all-masks visualization by default (with an opt-out switch).
- Client parser: accept JSON tool-call outputs that include a tool name and args.
- Remaining tool ports are deferred for now (not planned in the near term).
- OpenThinkIMG: `SegmentRegionAroundPoint` (SAM3 point-guided segmentation).
- VisualSketchpad: `segment_and_mark`, `sliding_window_detection`.
