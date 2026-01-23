# Researcher Tutorial: Subset Tool Server, LLM Tool Calls, and New Tools

This is a focused walkthrough for researchers who want to: (1) launch a tool server with a subset of tools and ask an LLM to predict a tool call, (2) parse + execute that call with the client, and (3) add new tools. Examples use `Python_Executor_Tool` because it is lightweight and does not require external model servers.

Run these steps from the tool infra root and set `PYTHONPATH="$(pwd)/src"` so imports resolve.

## Codebase walkthrough (quick map)

- `src/tools/`: tool packages; each tool lives in its own subdir with `tool.py` + `__init__.py`.
- `src/tools/base.py`: `BaseTool` metadata contract (`tool_name`, `input_types`, `output_type`).
- `src/tool_registry/loader.py`: loads tools from `TOOL_MODULES` (`auto` or comma-separated module list).
- `src/server/app.py`: FastAPI server with `/tools` and `/execute`.
- `src/client/`: `ToolClient`, `python_tool_call_parser.py` (Python tool-call parsing), `json_tool_call_parser.py` (JSON tool-call parsing).
- `src/llm/engine.py`: LLM wrapper used to generate tool-call code.
- `scripts/run_server.sh`: starts the tool server; requires `LIGHT_TOOL_MODULES` or `HEAVY_TOOL_MODULES` (builds `TOOL_MODULES`, ignore direct `TOOL_MODULES`).
- Tool model servers (heavy tools) live under their tool directories, for example:
  - `src/tools/sam3_segment/run_sam3_server.sh`
  - `src/tools/grounding_dino/run_grounding_dino_server.sh`
  - `src/tools/ocr/run_ocr_server.sh`
  - `src/tools/depth/run_depth_anything_server.sh`
- Subserver ports can be overridden via env vars: SAM3 uses `HOST`/`PORT`, GroundingDINO uses
  `GROUNDING_DINO_HOST`/`GROUNDING_DINO_PORT`, OCR uses `OCR_HOST`/`OCR_PORT`, Depth uses
  `DEPTH_ANYTHING_HOST`/`DEPTH_ANYTHING_PORT`. Use `python scripts/scan_server_status.py` to confirm
  tool/subtool servers are reachable.
- `LLM_Direct_Generator_Tool` reads `LLM_DIRECT_MODEL_BASE_URLS` (model:base_url) plus optional
  `LLM_DIRECT_MODEL_API_KEYS`/`LLM_DIRECT_API_KEY` (defaults to `EMPTY` for no auth). Base URLs
  may include `/v1`; if omitted, `/v1` is appended automatically.
- `LLM_Direct_Generator_Tool` is a heavy tool in the sense that you must launch the OpenAI-compatible
  model servers (for example vLLM) yourself and configure the `.env` mappings.

## 1) Start a server with a subset of tools + LLM predicts a tool call

Pick a subset of tools by listing their module paths in `LIGHT_TOOL_MODULES` (comma-separated). The module for `Python_Executor_Tool` is `tools.python_executor.tool`. Use `LIGHT_TOOL_MODULES=auto` to discover all `src/tools/*/tool.py` modules (auto skips tools that fail imports or sanity checks). When `LIGHT_TOOL_MODULES=auto`, `HEAVY_TOOL_MODULES` is ignored.

Before launching your own server, check whether one is already running so you can reuse it:
```bash
python scripts/scan_server_status.py
```
It scans ports 8000-9000 and prints tool lists plus any subtool servers it detects.

```bash
export LIGHT_TOOL_MODULES="tools.python_executor.tool"
./scripts/run_server.sh
```

Heavy tools: if your subset includes tools that depend on external model servers (SAM3, GroundingDINO, OCR, Depth), start those model servers first by running the corresponding `run_*_server.sh` inside the tool directory, using the correct conda environment for that model. The main tool server does not need those heavy dependencies installed as long as the model servers are already running.
Image tools accept `image_base64` for remote clients; use it instead of `image_path` when the tool server cannot see your local filesystem (use `base_image_base64`/`overlay_image_base64` for `OverlayImages_Tool`).


Example: ask an LLM to predict a tool call (the server needs to be running so you can grab tool metadata).

```python
import os

from client.client import ToolClient
from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from llm.engine import LLMEngine

client = ToolClient("http://127.0.0.1:8101")
metadata = client.list_tools()["Python_Executor_Tool"]

prompt = (
    "Generate a single Python code block that calls this tool.\n"
    "The code must include exactly one line: ToolName(...).\n"
    "Use the tool name as the function name.\n"
    "Use keyword arguments only.\n"
    f"Tool name: Python_Executor_Tool\n"
    f"Tool metadata: {metadata}\n"
    "User request: Print 7 + 9 and store it in a variable named value.\n"
    "Return only the code block."
)

engine = LLMEngine(
    model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
    base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
    api_key=os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY),
)

llm_output = engine.generate(prompt, max_tokens=8192, temperature=0.6)
print(llm_output)
```

## 2) Parse the tool call + execute via the parser + client

The Python tool-call parser expects a `ToolName(...)` call in the LLM output. JSON tool-call parsing accepts `{"tool_name": ..., "args": {...}}` or OpenAI-style `{"name": ..., "arguments": {...}}`.

```python
from client.client import ToolClient
from client.python_tool_call_parser import parse_python_tool_call

client = ToolClient("http://127.0.0.1:8101")
tool_call = parse_python_tool_call(llm_output)

response = client.execute(tool_call["tool_name"], tool_call["args"])
print(response["result"])
```

For JSON tool calls:

```python
from client.json_tool_call_parser import parse_json_tool_call

tool_call = parse_json_tool_call(llm_output_json)
response = client.execute(tool_call["tool_name"], tool_call["args"])
```

If the LLM output is missing required fields, insert guardrails before executing (for example, set a default `code` value).

## 3) Add a new tool (best done with a coding agent)

It is best to use a coding agent to scaffold the new tool files, metadata, and tests consistently. The typical flow:

1) Check relevant implementations in `reference/` and confirm expected inputs/outputs.
2) Create `src/tools/<tool_name>/tool.py` + `__init__.py`, implement a `_Tool` class that follows `BaseTool` metadata.
3) Add `tests/test_<tool_name>_tool.py` with direct, client, and LLM parsing paths.
4) Register the tool if you want explicit lists to include it:
   - Add the module to `.env.example` under `LIGHT_TOOL_MODULES` or `HEAVY_TOOL_MODULES`.
   - If you use `LIGHT_TOOL_MODULES=auto`, no list edits are needed.
5) Update `README.md` and `TUTORIAL.md` with any new env vars, API key files, or server requirements.

If the tool is heavy, add a `run_*_server.sh` under the tool directory and document the required conda environment and `*_SERVER_URL` env var for that tool.
