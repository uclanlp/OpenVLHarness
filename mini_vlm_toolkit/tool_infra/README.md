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
| `ClipSearch_Tool` | Search a video-clip vector database for events matching a textual description. |
| `FrameInspection_Tool` | Inspect frames from specified time ranges and answer detailed video questions. |
| `GlobalBrowse_Tool` | Retrieve relevant video clips from a vector database and synthesize a final answer. |
| `Gemini_Google_Search_Tool` | Gemini-backed Google search tool with optional citations. |
| `GroundingDINO_Tool` | Detect objects in an image using a GroundingDINO server and return bounding boxes. |
| `OCR_Tool` | Extract text strings from an image using an OCR server. |
| `OverlayImages_Tool` | Overlay an image onto another with alpha blending. |
| `Point_Tool` | Extract point coordinates from text or provided points and return normalized coordinates. |
| `Python_Code_Generator_Tool` | Generate and execute simple Python code snippets for basic arithmetic calculations and math-related problems. |
| `LLM_Direct_Generator_Tool` | Send OpenAI-compatible chat or embeddings requests to a configured server and return the full payload. |
| `Python_Executor_Tool` | Execute a provided Python code snippet and return its output and variables. |
| `SAM3_Segment_New_Tool` | Call the refactored SAM3 segmentation server and return RLE masks with a rendered overlay. |
| `SegmentAndMark_Tool` | Segment the full image and annotate each mask with a numeric label. |
| `SegmentRegionAroundPoint_Tool` | Segment a region around one or more points using SAM3. |
| `SerpApi_Image_Search_Tool` | SerpApi Google reverse-image search that returns top-5 similar-image matches for one image URL. |
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
./scripts/run_server.sh $ENV_FILE_NAME
```
`scripts/run_server.sh` sources the tool infra `.env` for
`HOST`, `PORT`, `LIGHT_TOOL_MODULES`, `HEAVY_TOOL_MODULES`, and `*_SERVER_URL` overrides
(main server only).
It builds `TOOL_MODULES` from `LIGHT_TOOL_MODULES` + `HEAVY_TOOL_MODULES`.
Do not set `TOOL_MODULES` directly when using `scripts/run_server.sh`.

Start the refactored SAM3 model server (required for `SAM3_Segment_New_Tool`):
```bash
./src/tools/sam3_segment_new/run_sam3_server.sh
```
Legacy SAM3 tool wrappers are kept under `legacy/sam3/` for reference and are not
discovered by `LIGHT_TOOL_MODULES=auto`.
Model server scripts live in their tool folders and do not read `.env`.
Use the tool-specific `*_PORT`/`*_HOST` env vars to change subserver ports (see Notes),
and run `python scripts/scan_server_status.py` to confirm they are up.

Start the OCR model server (required for the OCR tool):
```bash
./src/tools/ocr/run_ocr_server.sh
```

Start the GroundingDINO model server (required for the GroundingDINO tool):
```bash
./src/tools/grounding_dino/run_grounding_dino_server.sh
```

Start the Depth-Anything model server (required for the Depth tool):
```bash
./src/tools/depth/run_depth_anything_server.sh
```

Start the server with a specific tool list:
```bash
export LIGHT_TOOL_MODULES="tools.python_executor.tool"
export HEAVY_TOOL_MODULES="tools.sam3_segment_new.tool"
./scripts/run_server.sh
```
Exports override `.env` values for the same keys.
Set `LIGHT_TOOL_MODULES=auto` to discover all `src/tools/*/tool.py` modules.
Auto discovery skips tools that fail imports or sanity checks; explicit lists are strict.
When `LIGHT_TOOL_MODULES=auto`, `HEAVY_TOOL_MODULES` is ignored.
If set, `LIGHT_TOOL_MODULES` must be either `auto` or a list of module paths (no mixing).

## Planned Tool Design (VLM Mask Querying)

We plan to add a tool that queries each SAM3 mask with a VLM question, reusing
existing tool infra components:
- Name (proposed): `SAM3_QueryEachMask_VLM_Tool`
- Reuse `rendered_image_b64` from `SAM3_Segment_New_Tool` for mask-aware context.
- Reuse `LLM_Direct_Generator_Tool` to send OpenAI-compatible VLM requests.
- Parse `<answer>` tags (yes/no or free-form) and aggregate counts/answers.

This keeps the new tool lightweight and avoids importing the full grounding-agent
stack into tool_infra.

List tools:
```bash
curl http://127.0.0.1:8101/tools
```
List OpenAI-compatible tool schemas:
```bash
curl http://127.0.0.1:8101/jsonschema
```
Execute a tool (optional `tool_call_id` supported):
```bash
curl -X POST http://127.0.0.1:8101/execute \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"Python_Executor_Tool","args":{"code":"print(2+2)\nvalue=4"},"tool_call_id":"call_123"}'
```
`POST /execute` response contract:
- Success keeps legacy fields and adds envelope fields:
  - `ok: true`, `tool_name`, `result`, `error: null`, `tool_call_id`, `meta`
- Failure keeps HTTP non-2xx and returns:
  - `ok: false`, `tool_name`, `result: null`, `error`, `tool_call_id`, `meta`
Tool aliases:
- Tools can define `tool_aliases` in metadata (`tool.py`), and each tool also gets a generated snake_case alias.
- `GET /tools` exposes both canonical and alias entries.
  - Canonical entries include `is_alias: false`, `canonical_tool_name: <canonical>`, `tool_aliases: [...]`.
  - Alias entries include `is_alias: true`, `canonical_tool_name: <canonical>`.
- `GET /jsonschema` includes schemas for both canonical and alias names.
- `POST /execute` accepts canonical names or aliases; response `tool_name` always echoes the requested name.
- Alias matching is exact and case-sensitive.
- Alias names must be valid Python identifiers. Alias/canonical collisions fail startup.
- `GET /health` keeps `tool_count` and adds `canonical_tool_count` and `alias_tool_count`.
`scripts/run_server.sh` requires `LIGHT_TOOL_MODULES` or `HEAVY_TOOL_MODULES` (see `.env.example`).
If you run `uvicorn` directly, `TOOL_MODULES=auto` (or unset) triggers tool auto-discovery.

## CLI (no Python required)

Local execution (no server):
```bash
PYTHONPATH="$(pwd)/src" python -m cli --tool Python_Executor_Tool --code "print(2+2)\nvalue=2+2"
```

Server execution:
```bash
PYTHONPATH="$(pwd)/src" python -m cli --server http://127.0.0.1:8101 --tool Python_Executor_Tool --code "print(5*3)\nvalue=15"
```

## Tests

```bash
export PYTHONPATH="$(pwd)/src:$(pwd)"
python tests/test_python_executor.py
```

- Run all test scripts (prints a clean PASS/FAIL summary):
```bash
./scripts/run_tests.sh
```

- Run any tool test with `python tests/test_<tool>.py`.
- New lightweight transport/schema tests (no running tool server required):
```bash
python tests/test_execute_transport_contract.py
python tests/test_jsonschema_generation.py
python tests/test_tool_alias_registry.py
```
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
- Run commands from the tool infra root. For tests, set `PYTHONPATH="$(pwd)/src:$(pwd)"` so `tests` imports resolve.
- If you are using Di's shared server (always running on port 8101), use the Serper search, cloud search, and OpenAI-related tools with caution because they use Di's personal API key.
- `sam3_segment_new` also depends on `requirements-server.txt` (including `matplotlib`) and runs via `./src/tools/sam3_segment_new/run_sam3_server.sh`.
- Image tools accept base64 inputs for remote clients; pass exactly one of `image_path` or `image_base64` (use `base_image_base64`/`overlay_image_base64` for `OverlayImages_Tool`).
- GroundingDINO runs as a separate model server; set `GROUNDING_DINO_SERVER_URL` and launch `./src/tools/grounding_dino/run_grounding_dino_server.sh` (requires `torch`, `torchvision`, and `transformers`).
- OCR runs as a separate EasyOCR server; set `OCR_SERVER_URL` and launch `./src/tools/ocr/run_ocr_server.sh` (requires `easyocr`, downloads weights on first run, defaults to CPU).
- Depth runs as a separate Depth-Anything server; set `DEPTH_ANYTHING_SERVER_URL` and launch `./src/tools/depth/run_depth_anything_server.sh` (requires `torch` and `transformers`, optional `matplotlib` for Inferno colormap).
- Subserver ports can be overridden via env vars: SAM3 new uses `SAM3_NEW_HOST`/`SAM3_NEW_PORT`, GroundingDINO uses `GROUNDING_DINO_HOST`/`GROUNDING_DINO_PORT`, OCR uses `OCR_HOST`/`OCR_PORT`, Depth uses `DEPTH_ANYTHING_HOST`/`DEPTH_ANYTHING_PORT`.
- Heavy tool wrappers surface subserver error messages (status + detail) and truncate large payloads.
- Use `python scripts/scan_server_status.py` to confirm tool/subtool servers are reachable and report their health.
- `LLM_Direct_Generator_Tool` reads `LLM_DIRECT_MODEL_BASE_URLS` (model:base_url). Optional `LLM_DIRECT_MODEL_API_KEYS` and `LLM_DIRECT_API_KEY` default to `EMPTY` for no auth. Base URLs may include `/v1`; if omitted, `/v1` is appended automatically.
- Tools can now optionally define `input_schema` (JSON Schema for OpenAI-style tool calling). `/jsonschema` uses `input_schema` when present and falls back to inferred schema from `input_types` otherwise.
- Python tool-call parsing lives in `src/client/python_tool_call_parser.py` and expects `ToolName(...)` calls. JSON tool-call parsing lives in `src/client/json_tool_call_parser.py` and accepts `{"tool_name": ..., "args": {...}}` or OpenAI-style `{"name": ..., "arguments": {...}}`.
- API key files are read from:
  - `src/tools/gemini_google_search/google_api_key`
  - `src/tools/serpapi_image_search/serpapi_api_key`
  - `src/tools/serper_google_search/serper_api_key`
  - `src/tools/single_url_qa/openai_api_key`
- Reminder: run `python tests/test_gemini_google_search.py`, `python tests/test_serper_google_search.py`, `python tests/test_serpapi_image_search.py`, and `python tests/test_single_url_qa.py`.

## TODO
- Rui: adding a GUI grounder
- Client parser: add an optional LLM-based argument fixer for malformed tool calls.
- Tool onboarding: add a scaffolding helper for new tools.
- Tool metadata: clean up `/tools` `input_types` output (remove inline comments).
- Add explicit `input_schema` coverage for remaining non-SAM3 tools.
- Remaining tool ports are deferred for now (not planned in the near term).
  - OpenThinkIMG: `SegmentRegionAroundPoint` (SAM3 point-guided segmentation).
  - VisualSketchpad: `segment_and_mark`, `sliding_window_detection`.
