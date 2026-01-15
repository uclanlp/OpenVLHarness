# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python toolkit for tool execution and a FastAPI server. Key paths:
- `mini_vlm_toolkit/tools/`: Tool packages (each subdir has `tool.py` + `__init__.py`); heavy tools may also include `server.py`/`sam3_server.py` and a `README.md`.
- `mini_vlm_toolkit/tools/base.py`: `BaseTool` definition and metadata contract.
- `mini_vlm_toolkit/tools/image_utils.py`: Shared image helpers.
- `mini_vlm_toolkit/tool_registry/`: Tool loader and registry (`mini_vlm_toolkit/tool_registry/loader.py`).
- `mini_vlm_toolkit/server/`: FastAPI app (`mini_vlm_toolkit/server/app.py`).
- `mini_vlm_toolkit/client/`: HTTP client + tool-call parser.
- `mini_vlm_toolkit/llm/`: LLM engine wrapper used for tool-call generation.
- `mini_vlm_toolkit/tool_runner.py`: Minimal direct tool runner helper.
- `mini_vlm_toolkit/config.py`: Default LLM config values.
- `tests/`: Standalone Python test scripts + `tests/utils.py`.
- `tests/data`: Sample images (downloaded if missing).
- `tests/output`: Generated test images (gitignored).
- `scripts/`: Helper scripts (tool server + model servers).
- `README.md` and `TUTORIAL.md`: Usage docs.
- `reference/`: External reference materials; do not edit for project changes.

## Tool Registry & Launch Defaults
- `scripts/run_server.sh` requires `LIGHT_TOOL_MODULES` (`auto` or a comma-separated list) or `HEAVY_TOOL_MODULES`, and appends `HEAVY_TOOL_MODULES` only when `LIGHT_TOOL_MODULES` is explicit; it builds `TOOL_MODULES` and ignores direct `TOOL_MODULES`.
- `mini_vlm_toolkit/tool_registry/loader.py` uses `TOOL_MODULES`; `auto` (or unset) triggers tool auto-discovery that skips failed imports/sanity checks.
- Some modules define multiple tools (for example `mini_vlm_toolkit.tools.sam3_visualize.tool`).

## Build, Test, and Development Commands
- `pip install -r requirements-client.txt` installs runtime deps.
- `pip install -r requirements-server.txt` installs heavy model server deps (SAM3, GroundingDINO, OCR, Depth).
- `./scripts/run_server.sh` starts the FastAPI tool server (uses `HOST`/`PORT`, defaults to `127.0.0.1:8101`).
- `curl http://127.0.0.1:8101/tools` lists available tools.
- `PYTHONPATH="$(pwd)" python -m mini_vlm_toolkit.cli --tool Python_Executor_Tool --code "print(2+2)\nvalue=4"` runs a tool locally.
- `PYTHONPATH="$(pwd)" python -m mini_vlm_toolkit.cli --server http://127.0.0.1:8101 --tool Python_Executor_Tool --code "print(5*3)\nvalue=15"` runs via server.
- `PYTHONPATH="$(pwd)" python tests/test_python_executor.py` runs a test script.
- Model servers (run separately when needed):
  - `./mini_vlm_toolkit/tools/sam3_segment/run_sam3_server.sh`
  - `./mini_vlm_toolkit/tools/grounding_dino/run_grounding_dino_server.sh`
  - `./mini_vlm_toolkit/tools/ocr/run_ocr_server.sh`
  - `./mini_vlm_toolkit/tools/depth/run_depth_anything_server.sh`

## Adding a New Tool (Workflow)
1) Check relevant implementations in `reference/` and confirm the intended behavior/inputs/outputs before coding.
2) Create `mini_vlm_toolkit/tools/<tool_name>/tool.py` + `__init__.py`; follow `BaseTool` metadata conventions and keep inputs explicit.
3) Add tests in `tests/test_<tool_name>_tool.py` following the direct/client/LLM pattern; add any new images to `tests/data`.
4) Decide if the tool is heavy (external model server or large deps) or default (lightweight). If you use explicit tool lists, update `.env` to place the module in `LIGHT_TOOL_MODULES` or `HEAVY_TOOL_MODULES`. With `LIGHT_TOOL_MODULES=auto`, no list edits are needed.
5) Update docs (`README.md`, `TUTORIAL.md`) and note any new env vars or API key files.

## Coding Style & Naming Conventions
- Use 4-space indentation and standard PEP 8 conventions.
- Prefer `snake_case` for modules/functions and `PascalCase` for classes.
- Tool classes follow the `_Tool` suffix pattern (e.g., `Python_Executor_Tool`).
- Use uppercase module constants for tool IDs (e.g., `TOOL_NAME`).
- Keep tool metadata (`tool_name`, `input_types`, `output_type`) consistent with existing tools.
- No formatter/linter is configured; keep changes minimal and consistent with surrounding code.

## Testing Guidelines
- Tests are plain scripts that use `assert` (no pytest harness yet).
- Tool tests usually follow: direct invocation, client invocation via server, LLM-assisted invocation.
- Start the tool server before client/LLM tests or set `TOOL_SERVER_URL`.
- Vision tests download sample images into `tests/data` on first run and save outputs to `tests/output`.
- The LLM test requires `LLM_MODEL`; `LLM_BASE_URL` and `LLM_API_KEY` are optional.
- Web/URL QA tests require API keys present in the tool directories.
- SAM3/GroundingDINO/OCR/Depth tests require their respective model servers.

## Commit & Pull Request Guidelines
- Use concise, imperative commit summaries and include test results in the body when relevant.
- PRs should include a short summary, tests run, and any new env/config requirements.
- Call out tool metadata or tool-list changes explicitly.

## Configuration & Security Notes
- The Python executor runs `exec()` without sandboxing; avoid untrusted code.
- Tool lists are driven by `LIGHT_TOOL_MODULES` + `HEAVY_TOOL_MODULES`; see `.env.example` for sample lists.
- Environment variables commonly used: `HOST`, `PORT`, `LIGHT_TOOL_MODULES`, `HEAVY_TOOL_MODULES`, `TOOL_SERVER_URL`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`.
- Model server URLs:
  - `SAM3_SERVER_URL`
  - `GROUNDING_DINO_SERVER_URL`
  - `OCR_SERVER_URL` (languages configured by `OCR_LANGS` on the server; requests must match)
  - `DEPTH_ANYTHING_SERVER_URL` (colormap via `DEPTH_ANYTHING_COLORMAP`, optional `matplotlib`)
- API key files are read from:
  - `mini_vlm_toolkit/tools/gemini_google_search/google_api_key`
  - `mini_vlm_toolkit/tools/serper_google_search/serper_api_key`
  - `mini_vlm_toolkit/tools/single_url_qa/openai_api_key`
