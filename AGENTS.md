# Repository Guidelines

## Scope and Focus
This repo hosts two efforts:
- Tool infra (stable): `mini_vlm_toolkit/tool_infra` contains the lightweight tool server/client stack and is the primary development focus for this agent.
- Eval/benchmark stack (active + evolving): VLM evaluation code lives at the repo root and under `mini_vlm_toolkit/` (agents, datasets, inference utilities, configs, and `run_benchmark.py`). Conventions for that code are still evolving, so keep changes pragmatic and avoid over-enforcing style there.

Default to working inside `mini_vlm_toolkit/tool_infra` unless explicitly asked to touch eval code.

---

## Eval/Benchmark Infra (repo root + mini_vlm_toolkit)
- Entry point: `run_benchmark.py` loads `.env`, builds an agent from `configs/*.yaml`, runs inference, then calls dataset evaluation.
- Agent layer: `mini_vlm_toolkit/agent/` includes:
  - `BaseAgent` for direct OpenAI-compatible `/v1/chat/completions` requests.
  - `BaseAgentWithTools` for native `tool_calls` loops against tool server `/execute`.
  - Config loading/build (`load_agent_config`, `build_agent`) via YAML `agent_name` + `agent_kwargs`.
  - Module guide at `mini_vlm_toolkit/agent/README.md`.
- Dataset registry: `mini_vlm_toolkit/dataset/__init__.py` aggregates dataset classes and exposes `build_dataset`; video variants (frame/fps/pack) are defined in `mini_vlm_toolkit/dataset/video_dataset_config.py` and checked first.
- Dataset base classes: `mini_vlm_toolkit/dataset/image_base.py`, `mini_vlm_toolkit/dataset/video_base.py`, `mini_vlm_toolkit/dataset/text_base.py`.
- Inference utilities: `mini_vlm_toolkit/inference_image.py` and `mini_vlm_toolkit/inference_video.py` handle prompt building, API calls, and result file writing/merging.
- Shared utilities: `mini_vlm_toolkit/smp/` (IO helpers, data root, pred/eval formats) and `mini_vlm_toolkit/utils/` (progress + matching helpers).
- Data root and formats: set `LMUData` for dataset storage; optional `PRED_FORMAT`/`EVAL_FORMAT` control outputs.
- Agent examples/tests:
  - Examples: `examples/ad-hoc_ex_for_dev/base_agent_with_tools_chat_example.py`, `examples/ad-hoc_ex_for_dev/base_agent_with_tools_image_response_example.py`.
  - Tests: `mini_vlm_toolkit/agent/tests/test_base_agent_native_tool_call.py`, `mini_vlm_toolkit/agent/tests/test_tool_execute_request_envelope.py`.
- Common `BaseAgentWithTools` config knobs in YAML: `tool_base`, `tool_list`, `tool_choice`, `prompt_version`, `max_iterations`, and optional `extra_body`.
- Image handling in tool loops: runtime image IDs are exposed in system prompt, `*_image_path` args may be rewritten to `*_image_base64`, and unresolved references surface structured errors.

---

## Tool Infra (mini_vlm_toolkit/tool_infra)
All sections below apply only to tool infra. The eval/benchmark stack above is developed separately.

## Tool Infra Layout (mini_vlm_toolkit/tool_infra)
- `src/tools/`: Tool packages (each subdir has `tool.py` + `__init__.py`); heavy tools may also include `server.py`/`sam3_server.py` and a `README.md`.
- `src/tools/base.py`: `BaseTool` definition and metadata contract.
- `src/tools/image_utils.py`: Shared image helpers (path/base64 decoding + `resolve_image_input`).
- `src/tool_registry/`: Tool loader and registry (`src/tool_registry/loader.py`).
- `src/server/`: FastAPI app (`src/server/app.py`).
- `src/client/`: HTTP client + tool-call parsers (`python_tool_call_parser.py`, `json_tool_call_parser.py`).
- `src/llm/`: LLM engine wrapper used for tool-call generation.
- `src/tool_runner.py`: Minimal direct tool runner helper.
- `src/config.py`: Default LLM config values.
- `tests/`: Standalone Python test scripts + `tests/utils.py` (includes base64 encoding helpers for image tests).
- `tests/data`: Sample images (downloaded if missing).
- `tests/output`: Generated test images (gitignored).
- `scripts/`: Helper scripts (tool server, tests, status checks, model servers).
- `README.md`, `TUTORIAL.md`, `TUTORIAL_NEW.md`: Usage docs.
- `PROJECT_PLAN.md`: Project roadmap and TODO tracking.
- `reference/`: External reference materials; do not edit for project changes.

## Tool Registry & Launch Defaults
- `scripts/run_server.sh` expects the tool infra `.env`, logs its realpath, and exits if the file is missing.
- `scripts/run_server.sh` requires `LIGHT_TOOL_MODULES` (`auto` or a comma-separated list) or `HEAVY_TOOL_MODULES`, and appends `HEAVY_TOOL_MODULES` only when `LIGHT_TOOL_MODULES` is explicit; it builds `TOOL_MODULES` and ignores a user-set `TOOL_MODULES`.
- `src/tool_registry/loader.py` uses `TOOL_MODULES`; `auto` (or unset) triggers tool auto-discovery that skips failed imports/sanity checks.
- Subserver scripts for heavy tools live under their tool folders and do not load `.env`.
- Some modules define multiple tools (for example `tools.sam3_visualize.tool`).

## Build, Test, and Development Commands
Run these from `mini_vlm_toolkit/tool_infra`.
- `pip install -r requirements-client.txt` installs runtime deps.
- `pip install -r requirements-server.txt` installs heavy model server deps (SAM3, GroundingDINO, OCR, Depth).
- `./scripts/run_server.sh` starts the FastAPI tool server (uses `HOST`/`PORT`, defaults to `127.0.0.1:8101`).
- `curl http://127.0.0.1:8101/tools` lists available tools.
- `PYTHONPATH="$(pwd)/src" python -m cli --tool Python_Executor_Tool --code "print(2+2)\nvalue=4"` runs a tool locally.
- `PYTHONPATH="$(pwd)/src" python -m cli --server http://127.0.0.1:8101 --tool Python_Executor_Tool --code "print(5*3)\nvalue=15"` runs via server.
- `PYTHONPATH="$(pwd)/src:$(pwd)" python tests/test_python_executor.py` runs a test script.
- `./scripts/run_tests.sh` runs all tests sequentially with progress.
- `./scripts/scan_server_status.py` checks model server endpoints configured in `.env`.
- Model servers (run separately when needed):
  - `./src/tools/sam3_segment/run_sam3_server.sh`
  - `./src/tools/grounding_dino/run_grounding_dino_server.sh`
  - `./src/tools/ocr/run_ocr_server.sh`
  - `./src/tools/depth/run_depth_anything_server.sh`

## Adding a New Tool (Workflow)
1) Check relevant implementations in `reference/` and confirm the intended behavior/inputs/outputs before coding.
2) Create `src/tools/<tool_name>/tool.py` + `__init__.py`; follow `BaseTool` metadata conventions and keep inputs explicit.
   - If the tool accepts image inputs, add `image_base64` (or base/overlay variants) and enforce exactly one of path/base64.
3) Add tests in `tests/test_<tool_name>_tool.py` following the direct/client/LLM pattern; add any new images to `tests/data`.
4) Decide if the tool is heavy (external model server or large deps) or default (lightweight). If you use explicit tool lists, update `.env` to place the module in `LIGHT_TOOL_MODULES` or `HEAVY_TOOL_MODULES`. With `LIGHT_TOOL_MODULES=auto`, no list edits are needed.
5) Update docs (`README.md`, `TUTORIAL.md`) and note any new env vars or API key files.

## Tool Onboarding (Manual, No Scaffold)
- There is no tool scaffolding command; new tools are created by hand following the workflow above.
- Required files: `src/tools/<tool_name>/tool.py` and `src/tools/<tool_name>/__init__.py`.
- Recommended additions: `README.md` for heavy tools, plus `server.py`/`sam3_server.py` if a separate subserver is needed.
- Auto-discovery: with `LIGHT_TOOL_MODULES=auto`, new tool modules are picked up without editing any lists.
- Explicit lists: if you prefer explicit lists, update `.env` for `LIGHT_TOOL_MODULES`/`HEAVY_TOOL_MODULES` when adding tools.

## Tool Call Formats
- Python code tool-call parser expects `Tool_Name(arg=value, ...)` (tool name is required).
- JSON tool-call parser accepts either `{\"tool_name\": \"...\", \"args\": {...}}` or OpenAI-style `{\"name\": \"...\", \"arguments\": {...}}`.
- Use the JSON parser for OpenAI-style tool call payloads or when the LLM emits JSON directly.

## Tool Aliases
- Define aliases in each tool module via `BaseTool(..., tool_aliases=[...])`; keep alias logic in `mini_vlm_toolkit/tool_infra`.
- Alias names must be valid Python identifiers; matching is exact and case-sensitive.
- Server metadata always exposes canonical names and aliases in both `/tools` and `/jsonschema`.
- Alias collisions (alias vs canonical name or alias vs alias) are startup errors; fail fast during tool registry loading.
- `/execute` accepts canonical names or aliases, and returns the originally requested name in response `tool_name`.
- Alias metadata fields are `is_alias` and `canonical_tool_name`; canonical metadata keeps `tool_aliases`.
- The registry also auto-adds snake_case aliases for canonical tool names; avoid explicit aliases that would collide.

## Image Input Conventions
- Image tools accept either `image_path` (server-local path) or `image_base64` (base64 string or data URL); require exactly one.
- `OverlayImages_Tool` uses `base_image_base64`/`overlay_image_base64` in place of paths.
- For PIL-based tools, prefer `resolve_image_input` from `src/tools/image_utils.py`.
- For multipart server tools, use `decode_base64_image_data` and send bytes as multipart files.

## Coding Style & Naming Conventions
- Use 4-space indentation and standard PEP 8 conventions.
- Prefer `snake_case` for modules/functions and `PascalCase` for classes.
- Tool classes follow the `_Tool` suffix pattern (e.g., `Python_Executor_Tool`).
- Use uppercase module constants for tool IDs (e.g., `TOOL_NAME`).
- Keep tool metadata (`tool_name`, `input_types`, `output_type`) consistent with existing tools.
- No formatter/linter is configured; keep changes minimal and consistent with surrounding code.

## Testing Guidelines
- Tests are plain scripts that use `assert` (no pytest harness yet).
- Tool tests usually follow: direct invocation, client invocation via server, LLM-assisted invocation. Image tools also include base64 variants.
- Start the tool server before client/LLM tests or set `TOOL_SERVER_URL`.
- Vision tests download sample images into `tests/data` on first run and save outputs to `tests/output`.
- The LLM test requires `LLM_MODEL`; `LLM_BASE_URL` and `LLM_API_KEY` are optional.
- LLM tool tests cover both Python and JSON tool-call parsing paths.
- Web/URL QA tests require API keys present in the tool directories.
- SAM3/GroundingDINO/OCR/Depth tests require their respective model servers.

## Configuration & Security Notes
- The Python executor runs `exec()` without sandboxing; avoid untrusted code.
- Tool lists are driven by `LIGHT_TOOL_MODULES` + `HEAVY_TOOL_MODULES`; see `.env.example` for sample lists.
- Environment variables commonly used: `HOST`, `PORT`, `LIGHT_TOOL_MODULES`, `HEAVY_TOOL_MODULES`, `TOOL_SERVER_URL`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_DIRECT_MODEL_BASE_URLS`, `LLM_DIRECT_MODEL_API_KEYS`, `LLM_DIRECT_API_KEY`.
- `LLM_DIRECT_MODEL_BASE_URLS` maps `model:base_url` pairs (comma-separated). `LLM_DIRECT_MODEL_API_KEYS` can supply per-model keys, with `LLM_DIRECT_API_KEY` as the default.
- Model server URLs:
  - `SAM3_SERVER_URL`
  - `GROUNDING_DINO_SERVER_URL`
  - `OCR_SERVER_URL` (languages configured by `OCR_LANGS` on the server; requests must match)
  - `DEPTH_ANYTHING_SERVER_URL` (colormap via `DEPTH_ANYTHING_COLORMAP`, optional `matplotlib`)
- API key files are read from:
  - `src/tools/gemini_google_search/google_api_key`
  - `src/tools/serper_google_search/serper_api_key`
  - `src/tools/single_url_qa/openai_api_key`
