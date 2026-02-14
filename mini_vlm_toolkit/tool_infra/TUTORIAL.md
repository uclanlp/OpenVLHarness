# Tool Server + Client Tutorial

This tutorial is for colleagues training vision-language models to generate tool calls. It focuses on endpoint layout and client usage (not server/client internals).

## Prereqs

Install dependencies and start the server (install torch first). You can ask Di to start a server and forward to your machine as well. If you are using Di's shared server (always running on port 8101), use the Serper search, cloud search, and OpenAI-related tools with caution because they use Di's personal API key.

Run these steps from the tool infra root and set `PYTHONPATH="$(pwd)/src"` so imports resolve.

Before launching your own server, check whether one is already running so you can reuse it:
```bash
python scripts/scan_server_status.py
```
It scans ports 8000-9000 and prints tools on each tool server plus any subtool servers it detects.
```bash
pip install -r requirements-server.txt
cp .env.example .env
# Edit .env to set LIGHT_TOOL_MODULES (use "auto" to discover tools) and/or HEAVY_TOOL_MODULES.
./scripts/run_server.sh
```
Exports in your shell override `.env` values for the same keys.
Auto discovery skips tools that fail imports or sanity checks; explicit lists are strict.
When `LIGHT_TOOL_MODULES=auto`, `HEAVY_TOOL_MODULES` is ignored.
If set, `LIGHT_TOOL_MODULES` must be either `auto` or a list of module paths (no mixing).
Do not set `TOOL_MODULES` directly when using `scripts/run_server.sh`.

By default the server listens at `http://127.0.0.1:8101`.

If you don't need to start the server on your end, you can install the lighter `requirements-client.txt`.

Subserver ports are configurable via env vars (SAM3 new uses `SAM3_NEW_HOST`/`SAM3_NEW_PORT`,
GroundingDINO uses `GROUNDING_DINO_HOST`/`GROUNDING_DINO_PORT`, OCR uses `OCR_HOST`/`OCR_PORT`,
Depth uses `DEPTH_ANYTHING_HOST`/`DEPTH_ANYTHING_PORT`). After launching, run
`python scripts/scan_server_status.py` to confirm tool/subtool servers are reachable.

`LLM_Direct_Generator_Tool` reads `LLM_DIRECT_MODEL_BASE_URLS` (model:base_url) and optional
`LLM_DIRECT_MODEL_API_KEYS`/`LLM_DIRECT_API_KEY` (defaults to `EMPTY` for no auth). Base URLs
may include `/v1`; if omitted, `/v1` is appended automatically.
It is a heavy tool in the sense that you must launch the OpenAI-compatible model servers (for example
vLLM) yourself and configure the `.env` mappings.


## Endpoint layout

The FastAPI server exposes three main endpoints:

- `GET /tools` returns a JSON dict keyed by exposed tool name (canonical names and aliases).
- `GET /jsonschema` returns OpenAI-compatible function-tool schemas.
- `POST /execute` accepts `{"tool_name": "...", "args": {...}, "tool_call_id": "optional"}`.
  - Success response: `{"ok": true, "tool_name": "...", "result": ..., "error": null, "tool_call_id": "...", "meta": {...}}`
  - Failure response (non-2xx): `{"ok": false, "tool_name": "...", "result": null, "error": {...}, "tool_call_id": "...", "meta": {...}}`

Alias behavior:
- Canonical entries in `/tools` include `tool_aliases`, `is_alias: false`, and `canonical_tool_name`.
- Alias entries include `is_alias: true` and `canonical_tool_name` pointing to the canonical tool.
- `/jsonschema` includes schemas for both canonical names and aliases.
- `/execute` accepts canonical names or aliases, and echoes the exact requested `tool_name`.
- Alias matching is exact and case-sensitive.
- `/health` reports `tool_count`, `canonical_tool_count`, and `alias_tool_count`.

Quick check:
```bash
curl http://127.0.0.1:8101/tools
curl http://127.0.0.1:8101/jsonschema
curl -X POST http://127.0.0.1:8101/execute \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"Python_Executor_Tool","args":{"code":"print(2+2)\nvalue=4"},"tool_call_id":"call_123"}'
```

## Example 1: Python executor (no vision)

### Client-only call
```python
from client.client import ToolClient

client = ToolClient("http://127.0.0.1:8101")
response = client.execute(
    "Python_Executor_Tool",
    {"code": "print(2 + 2)\nvalue = 2 + 2"},
)
print(response["result"])
```

### Call from an LLM (generate tool call, parse, execute)
```python
import os

from client.client import ToolClient
from client.python_tool_call_parser import parse_python_tool_call
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
llm_output = engine.generate(prompt, max_tokens=1024, temperature=0.4)
tool_call = parse_python_tool_call(llm_output)

response = client.execute(tool_call["tool_name"], tool_call["args"])
print(response["result"])
```

## Example 2: ZoomInImageByBBox tool (vision input)

`ZoomInImageByBBox_Tool` crops an image using a normalized bbox and optional padding. It returns a base64-encoded cropped image. Image tools accept either `image_path` or `image_base64` (exactly one); `OverlayImages_Tool` uses `base_image_base64`/`overlay_image_base64` instead of paths.

### Client-only call
```python
from client.client import ToolClient

client = ToolClient("http://127.0.0.1:8101")
image_path = "tests/data/vision_tools_demo.jpg"

response = client.execute(
    "ZoomInImageByBBox_Tool",
    {
        "image_path": image_path,
        "bbox": [0.1, 0.1, 0.4, 0.4],
        "padding": 0.05,
    },
)
print(response["result"])  # {'cropped_image_base64': '...'}
```

If the tool server cannot access your filesystem, pass `image_base64` instead:
```python
import base64

with open(image_path, "rb") as handle:
    image_base64 = base64.b64encode(handle.read()).decode("utf-8")

response = client.execute(
    "ZoomInImageByBBox_Tool",
    {
        "image_base64": image_base64,
        "bbox": [0.1, 0.1, 0.4, 0.4],
        "padding": 0.05,
    },
)
print(response["result"])
```

### Call from an LLM (generate tool call, parse, execute)
```python
import os

from client.client import ToolClient
from client.python_tool_call_parser import parse_python_tool_call
from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from llm.engine import LLMEngine

client = ToolClient("http://127.0.0.1:8101")
metadata = client.list_tools()["ZoomInImageByBBox_Tool"]
image_path = "tests/data/vision_tools_demo.jpg"

prompt = (
    "Generate a single Python code block that calls this tool.\n"
    "The code must include exactly one line: ToolName(...).\n"
    "Use the tool name as the function name.\n"
    "Use keyword arguments only.\n"
    f"Tool name: ZoomInImageByBBox_Tool\n"
    f"Tool metadata: {metadata}\n"
    f"Image path to use: {image_path}\n"
    "User request: Zoom into bbox [0.05, 0.05, 0.45, 0.45] with padding 0.05.\n"
    "Return only the code block."
)

engine = LLMEngine(
    model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
    base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
    api_key=os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY),
)
llm_output = engine.generate(prompt, max_tokens=1024, temperature=0.4)
tool_call = parse_python_tool_call(llm_output)
args = tool_call["args"]

# Guardrails if the LLM omitted required fields.
args.setdefault("image_path", image_path)
args.setdefault("bbox", [0.05, 0.05, 0.45, 0.45])
args.setdefault("padding", 0.05)

response = client.execute(tool_call["tool_name"], args)
print(response["result"])
```

## Example 3: SAM3 segment new tool (vision input)

`SAM3_Segment_New_Tool` calls the refactored SAM3 server. Make sure the SAM3 server is running and reachable via `SAM3_NEW_SERVER_URL` (default `http://127.0.0.1:8086`).

### Client-only call
```python
from client.client import ToolClient

client = ToolClient("http://127.0.0.1:8101")
image_path = "tests/data/sam3_demo.jpg"

response = client.execute(
    "SAM3_Segment_New_Tool",
    {
        "prompt": "shoe",
        "image_path": image_path,
    },
)
print(response["result"])
```
If the tool server runs on another machine, send the image bytes instead of a server-local path:
```python
import base64

from client.client import ToolClient

client = ToolClient("http://127.0.0.1:8101")
image_path = "tests/data/sam3_demo.jpg"
with open(image_path, "rb") as handle:
    image_base64 = base64.b64encode(handle.read()).decode("utf-8")

response = client.execute(
    "SAM3_Segment_New_Tool",
    {
        "prompt": "shoe",
        "image_base64": image_base64,
    },
)
print(response["result"])
```
`SAM3_Segment_New_Tool` returns `pred_boxes`, `pred_masks` (RLE), `pred_scores`, and
`rendered_image_b64`.
The legacy names `SAM3_Segment_Tool` and `sam3_segment` are supported aliases.

### Call from an LLM (generate tool call, parse, execute)
```python
import os

from client.client import ToolClient
from client.python_tool_call_parser import parse_python_tool_call
from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from llm.engine import LLMEngine

client = ToolClient("http://127.0.0.1:8101")
metadata = client.list_tools()["SAM3_Segment_New_Tool"]
image_path = "tests/data/sam3_demo.jpg"

prompt = (
    "Generate a single Python code block that calls this tool.\n"
    "The code must include exactly one line: ToolName(...).\n"
    "Use the tool name as the function name.\n"
    "Use keyword arguments only.\n"
    f"Tool name: SAM3_Segment_New_Tool\n"
    f"Tool metadata: {metadata}\n"
    f"Image path to use: {image_path}\n"
    "User request: Segment the shoes in the image.\n"
    "Return only the code block."
)

engine = LLMEngine(
    model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
    base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
    api_key=os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY),
)
llm_output = engine.generate(prompt, max_tokens=8192, temperature=0.4)
tool_call = parse_python_tool_call(llm_output)
args = tool_call["args"]

# Guardrails if the LLM omitted required fields.
args.setdefault("prompt", "shoe")
args.setdefault("image_path", image_path)

response = client.execute(tool_call["tool_name"], args)
print(response["result"])
```

## Example 4: Qwen3 embeddings via LLM_Direct_Generator_Tool

Make sure your vLLM server is running and `.env` maps the embedding model name to its base URL
(see `src/tools/llm_direct_generator/README.md`).

```python
from client.client import ToolClient

client = ToolClient("http://127.0.0.1:8101")

response = client.execute(
    "LLM_Direct_Generator_Tool",
    {
        "model": "Qwen/Qwen3-Embedding-8B",
        "request_type": "embeddings",
        "input": [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other.",
        ],
    },
)
print(response["result"]["data"][0]["embedding"][:8])
```

## Notes for VLM tool-call training

- The server only sees the `tool_name` and `args`. If your model needs image context, handle that on the model side and pass either `image_path` or `image_base64` through the tool call (use `base_image_base64`/`overlay_image_base64` for overlays).
- The Python tool-call parser expects a `ToolName(...)` call in code. JSON tool-call parsing (`src/client/json_tool_call_parser.py`) accepts `{"tool_name": ..., "args": {...}}` or OpenAI-style `{"name": ..., "arguments": {...}}`.
- You can use `client.list_tools()` to capture tool metadata for training prompts or schema conditioning.
