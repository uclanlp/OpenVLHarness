# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python toolkit for tool execution and a FastAPI server. Key paths:
- `mini_vlm_toolkit/client/`: HTTP client and LLM tool-call helpers.
- `mini_vlm_toolkit/server/`: FastAPI app (`mini_vlm_toolkit/server/app.py`) and entrypoint for the tool server.
- `mini_vlm_toolkit/tools/`: Tool implementations grouped by tool name (each subdir has `tool.py`).
- `mini_vlm_toolkit/tool_registry/`: Loader that registers tools for the server.
- `mini_vlm_toolkit/llm/`: LLM engine wrapper used for tool-call generation.
- `tests/`: Standalone Python test scripts.
- `scripts/`: Helper scripts such as `scripts/run_server.sh`.
- `mini_vlm_toolkit/cli.py`: CLI entrypoint for local or server execution.
- `reference/`: External reference materials; do not edit for project changes.

## Build, Test, and Development Commands
- `pip install -r requirements-client.txt` installs runtime dependencies.
- `pip install -r requirements-server.txt` installs heavy model server dependencies (SAM3, GroundingDINO, OCR, Depth).
- `./scripts/run_server.sh` starts the FastAPI tool server (uses `HOST`/`PORT`, defaults to `127.0.0.1:8101`).
- `curl http://127.0.0.1:8101/tools` lists available tools.
- `PYTHONPATH="$(pwd)" python -m mini_vlm_toolkit.cli --tool Python_Executor_Tool --code "print(2+2)\nvalue=4"` runs a tool locally; keep `PYTHONPATH` set so `mini_vlm_toolkit.*` imports resolve.
- `PYTHONPATH="$(pwd)" python tests/test_python_executor.py` runs the current test script.
- Model servers (run separately when needed):
  - `./scripts/run_sam3_server.sh`
  - `./scripts/run_grounding_dino_server.sh`
  - `./scripts/run_ocr_server.sh`
  - `./scripts/run_depth_anything_server.sh`

## Coding Style & Naming Conventions
- Use 4-space indentation and standard PEP 8 conventions.
- Prefer `snake_case` for functions/modules and `PascalCase` classes; tool classes follow the `_Tool` suffix pattern (e.g., `Python_Executor_Tool`).
- Use uppercase module constants for tool IDs (e.g., `TOOL_NAME`).
- No formatter/linter is configured; keep changes minimal and consistent with surrounding code.

## Testing Guidelines
- Tests are simple scripts that use `assert` (no pytest harness yet).
- Tool tests follow the pattern: direct invocation, client invocation via server, LLM-assisted invocation.
- Start the tool server before client/LLM tests or set `TOOL_SERVER_URL`.
- Vision tests download sample images into `tests/data` on first run and save outputs to `tests/output` with timestamps (gitignored).
- The LLM test requires `LLM_MODEL`; `LLM_BASE_URL` and `LLM_API_KEY` are optional.

## Commit & Pull Request Guidelines
- Git history only contains “Initial commit,” so no established commit format exists.
- Use concise, imperative commit summaries and include test results in the body when relevant.
- PRs should include a short summary, tests run, and any new env/config requirements; note tool metadata changes explicitly.

## Configuration & Security Notes
- The Python executor runs `exec()` without sandboxing; avoid untrusted code.
- Heavy tools are exposed as wrappers around standalone model servers; they do not run GPU code in the tool server process.
- Environment variables commonly used: `HOST`, `PORT`, `TOOL_SERVER_URL`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`.
- Model server URLs:
  - `SAM3_SERVER_URL`
  - `GROUNDING_DINO_SERVER_URL`
  - `OCR_SERVER_URL` (languages configured by `OCR_LANGS` on the server; requests must match)
  - `DEPTH_ANYTHING_SERVER_URL` (colormap via `DEPTH_ANYTHING_COLORMAP`, optional `matplotlib`)
