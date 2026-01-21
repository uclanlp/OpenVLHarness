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
| `LLM_Direct_Generator_Tool` | Send OpenAI-compatible chat or embeddings requests to a configured server and return the full payload. |
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

Instead of using the quick start below, you are better off checking `TUTORIAL_NEW.md` and `TUTORIAL.md` for the guidance.

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
`LLM_Direct_Generator_Tool` is also a heavy tool in the sense that it depends on an external
OpenAI-compatible model server (for example vLLM), but it does not require `requirements-server.txt`.

Before launching your own server, check whether one is already running so you can reuse it:
```bash
python scripts/scan_server_status.py
```
The helper scans ports 8000-9000 and prints tool-server tool lists and any subtool servers it finds.

Start the server:
```bash
cp .env.example .env
# Edit .env to set LIGHT_TOOL_MODULES (use "auto" to discover tools) and/or HEAVY_TOOL_MODULES.
./scripts/run_server.sh
```
`scripts/run_server.sh` sources the repo-root `.env` for
`HOST`, `PORT`, `LIGHT_TOOL_MODULES`, `HEAVY_TOOL_MODULES`, and `*_SERVER_URL` overrides
(main server only).
It builds `TOOL_MODULES` from `LIGHT_TOOL_MODULES` + `HEAVY_TOOL_MODULES`.
Do not set `TOOL_MODULES` directly when using `scripts/run_server.sh`.

Start the SAM3 model server (required for the SAM3 tool):
```bash
./mini_vlm_toolkit/tools/sam3_segment/run_sam3_server.sh
```
Model server scripts live in their tool folders and do not read `.env`.
Use the tool-specific `*_PORT`/`*_HOST` env vars to change subserver ports (see Notes),
and run `python scripts/scan_server_status.py` to confirm they are up.

Start the OCR model server (required for the OCR tool):
```bash
./mini_vlm_toolkit/tools/ocr/run_ocr_server.sh
```

Start the GroundingDINO model server (required for the GroundingDINO tool):
```bash
./mini_vlm_toolkit/tools/grounding_dino/run_grounding_dino_server.sh
```

Start the Depth-Anything model server (required for the Depth tool):
```bash
./mini_vlm_toolkit/tools/depth/run_depth_anything_server.sh
```

Start the server with a specific tool list:
```bash
export LIGHT_TOOL_MODULES="mini_vlm_toolkit.tools.python_executor.tool,mini_vlm_toolkit.tools.sam3_visualize.tool"
export HEAVY_TOOL_MODULES="mini_vlm_toolkit.tools.sam3_segment.tool"
./scripts/run_server.sh
```
Exports override `.env` values for the same keys.
Set `LIGHT_TOOL_MODULES=auto` to discover all `mini_vlm_toolkit/tools/*/tool.py` modules.
Auto discovery skips tools that fail imports or sanity checks; explicit lists are strict.
When `LIGHT_TOOL_MODULES=auto`, `HEAVY_TOOL_MODULES` is ignored.
If set, `LIGHT_TOOL_MODULES` must be either `auto` or a list of module paths (no mixing).
Note: `mini_vlm_toolkit.tools.sam3_visualize.tool` defines two tools:
`SAM3_VisualizeAllMasks_Tool` and `SAM3_VisualizeSingleMask_Tool`.

List tools:
```bash
curl http://127.0.0.1:8101/tools
```
`scripts/run_server.sh` requires `LIGHT_TOOL_MODULES` or `HEAVY_TOOL_MODULES` (see `.env.example`).
If you run `uvicorn` directly, `TOOL_MODULES=auto` (or unset) triggers tool auto-discovery.

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

- Run all test scripts (prints a clean PASS/FAIL summary):
```bash
./scripts/run_tests.sh
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

Example LLM direct generator env:
```bash
export LLM_DIRECT_MODEL_BASE_URLS=Qwen/Qwen3-8B:http://localhost:8004/v1
# Optional: per-model API keys (defaults to EMPTY = no auth)
export LLM_DIRECT_MODEL_API_KEYS=Qwen/Qwen3-8B:EMPTY
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
- GroundingDINO runs as a separate model server; set `GROUNDING_DINO_SERVER_URL` and launch `./mini_vlm_toolkit/tools/grounding_dino/run_grounding_dino_server.sh` (requires `torch`, `torchvision`, and `transformers`).
- OCR runs as a separate EasyOCR server; set `OCR_SERVER_URL` and launch `./mini_vlm_toolkit/tools/ocr/run_ocr_server.sh` (requires `easyocr`, downloads weights on first run, defaults to CPU).
- Depth runs as a separate Depth-Anything server; set `DEPTH_ANYTHING_SERVER_URL` and launch `./mini_vlm_toolkit/tools/depth/run_depth_anything_server.sh` (requires `torch` and `transformers`, optional `matplotlib` for Inferno colormap).
- Subserver ports can be overridden via env vars: SAM3 uses `HOST`/`PORT`, GroundingDINO uses `GROUNDING_DINO_HOST`/`GROUNDING_DINO_PORT`, OCR uses `OCR_HOST`/`OCR_PORT`, Depth uses `DEPTH_ANYTHING_HOST`/`DEPTH_ANYTHING_PORT`.
- Use `python scripts/scan_server_status.py` to confirm tool/subtool servers are reachable and report their health.
- `LLM_Direct_Generator_Tool` reads `LLM_DIRECT_MODEL_BASE_URLS` (model:base_url). Optional `LLM_DIRECT_MODEL_API_KEYS` and `LLM_DIRECT_API_KEY` default to `EMPTY` for no auth. Base URLs may include `/v1`; if omitted, `/v1` is appended automatically.
- Python tool-call parsing lives in `mini_vlm_toolkit/client/python_tool_call_parser.py` and expects `ToolName(...)` calls. JSON tool-call parsing lives in `mini_vlm_toolkit/client/json_tool_call_parser.py` and accepts `{"tool_name": ..., "args": {...}}` or OpenAI-style `{"name": ..., "arguments": {...}}`.
- API key files are read from:
  - `mini_vlm_toolkit/tools/gemini_google_search/google_api_key`
  - `mini_vlm_toolkit/tools/serper_google_search/serper_api_key`
  - `mini_vlm_toolkit/tools/single_url_qa/openai_api_key`
- Reminder: run `python tests/test_gemini_google_search.py`, `python tests/test_serper_google_search.py`, and `python tests/test_single_url_qa.py`.
- SAM3 visualize tests: `python tests/test_sam3_visualize.py` (requires SAM3 + tool server) or `python tests/test_sam3_visualize_synthetic.py` (tool server only).

## TODO
- Fix error propagation for heavy tools: when subserver calls fail, capture response body/JSON and surface it in tool exceptions (include status + message, truncate large payloads).
- Rui: adding a GUI grounder
- Client parser: add an optional LLM-based argument fixer for malformed tool calls.
- Tool onboarding: add a scaffolding helper for new tools.
- Server error payloads: return structured error details (client now surfaces `detail`).
- Remaining tool ports are deferred for now (not planned in the near term).
  - OpenThinkIMG: `SegmentRegionAroundPoint` (SAM3 point-guided segmentation).
  - VisualSketchpad: `segment_and_mark`, `sliding_window_detection`.
