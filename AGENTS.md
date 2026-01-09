# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python toolkit for tool execution and a FastAPI server. Key paths:
- `client/`: HTTP client and LLM tool-call helpers.
- `server/`: FastAPI app (`server/app.py`) and entrypoint for the tool server.
- `tools/`: Tool implementations grouped by tool name (each subdir has `tool.py`).
- `tool_registry/`: Loader that registers tools for the server.
- `llm/`: LLM engine wrapper used for tool-call generation.
- `tests/`: Standalone Python test scripts.
- `scripts/`: Helper scripts such as `scripts/run_server.sh`.
- `cli.py`: CLI entrypoint for local or server execution.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs runtime dependencies.
- `./scripts/run_server.sh` starts the FastAPI server (uses `HOST`/`PORT`, defaults to `127.0.0.1:8001`).
- `curl http://127.0.0.1:8001/tools` lists available tools.
- `PYTHONPATH="$(pwd)" python cli.py --tool Python_Executor_Tool --code "print(2+2)\nvalue=4"` runs a tool locally; keep `PYTHONPATH` set so `mini_vlm_tool_kit.*` imports resolve.
- `PYTHONPATH="$(pwd)" python tests/test_python_executor.py` runs the current test script.

## Coding Style & Naming Conventions
- Use 4-space indentation and standard PEP 8 conventions.
- Prefer `snake_case` for functions/modules and `PascalCase` classes; tool classes follow the `_Tool` suffix pattern (e.g., `Python_Executor_Tool`).
- Use uppercase module constants for tool IDs (e.g., `TOOL_NAME`).
- No formatter/linter is configured; keep changes minimal and consistent with surrounding code.

## Testing Guidelines
- Tests are simple scripts that use `assert` (no pytest harness yet).
- `tests/test_python_executor.py` covers direct, client, and LLM-assisted execution.
- Start the server before client/LLM tests or set `TOOL_SERVER_URL`.
- The LLM test requires `LLM_MODEL`; `LLM_BASE_URL` and `LLM_API_KEY` are optional.

## Commit & Pull Request Guidelines
- Git history only contains “Initial commit,” so no established commit format exists.
- Use concise, imperative commit summaries and include test results in the body when relevant.
- PRs should include a short summary, tests run, and any new env/config requirements; note tool metadata changes explicitly.

## Configuration & Security Notes
- The Python executor runs `exec()` without sandboxing; avoid untrusted code.
- Environment variables commonly used: `HOST`, `PORT`, `TOOL_SERVER_URL`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`.
