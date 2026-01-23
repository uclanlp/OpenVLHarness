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

Subserver ports are configurable via env vars (SAM3 uses `HOST`/`PORT`, GroundingDINO uses
`GROUNDING_DINO_HOST`/`GROUNDING_DINO_PORT`, OCR uses `OCR_HOST`/`OCR_PORT`, Depth uses
`DEPTH_ANYTHING_HOST`/`DEPTH_ANYTHING_PORT`). After launching, run
`python scripts/scan_server_status.py` to confirm tool/subtool servers are reachable.

`LLM_Direct_Generator_Tool` reads `LLM_DIRECT_MODEL_BASE_URLS` (model:base_url) and optional
`LLM_DIRECT_MODEL_API_KEYS`/`LLM_DIRECT_API_KEY` (defaults to `EMPTY` for no auth). Base URLs
may include `/v1`; if omitted, `/v1` is appended automatically.
It is a heavy tool in the sense that you must launch the OpenAI-compatible model servers (for example
vLLM) yourself and configure the `.env` mappings.


## Endpoint layout

The FastAPI server exposes two main endpoints:

- `GET /tools` returns a JSON dict keyed by tool name.
- `POST /execute` accepts `{"tool_name": "...", "args": {...}}` and returns `{"tool_name": ..., "result": ...}`.

Quick check:
```bash
curl http://127.0.0.1:8101/tools
curl -X POST http://127.0.0.1:8101/execute \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"Python_Executor_Tool","args":{"code":"print(2+2)\nvalue=4"}}'
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

## Example 3: SAM3 segment tool (vision input)

`SAM3_Segment_Tool` calls an external SAM3 server. Make sure the SAM3 server is running and reachable via `SAM3_SERVER_URL` (default `http://127.0.0.1:8083`).

### Client-only call
```python
from client.client import ToolClient

client = ToolClient("http://127.0.0.1:8101")
image_path = "tests/data/sam3_demo.jpg"

response = client.execute(
    "SAM3_Segment_Tool",
    {
        "prompt": "shoe",
        "image_path": image_path,
        "return_masks_png_b64": True,
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
    "SAM3_Segment_Tool",
    {
        "prompt": "shoe",
        "image_base64": image_base64,
        "return_masks_png_b64": True,
    },
)
print(response["result"])
```
`SAM3_Segment_Tool` returns `all_masks_viz_base64` by default (overlay of all masks).
Set `return_all_masks_viz=False` to skip generating the visualization.

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
metadata = client.list_tools()["SAM3_Segment_Tool"]
image_path = "tests/data/sam3_demo.jpg"

prompt = (
    "Generate a single Python code block that calls this tool.\n"
    "The code must include exactly one line: ToolName(...).\n"
    "Use the tool name as the function name.\n"
    "Use keyword arguments only.\n"
    f"Tool name: SAM3_Segment_Tool\n"
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
args.setdefault("return_masks_png_b64", True)
args.setdefault("return_all_masks_viz", True)

response = client.execute(tool_call["tool_name"], args)
print(response["result"])
```

### Visualize SAM3 masks
```python
from client.client import ToolClient

client = ToolClient("http://127.0.0.1:8101")
image_path = "tests/data/sam3_demo.jpg"

sam3_response = client.execute(
    "SAM3_Segment_Tool",
    {
        "prompt": "shoe",
        "image_path": image_path,
        "return_masks_png_b64": True,
    },
)

print(sam3_response["result"]["all_masks_viz_base64"])

viz_all = client.execute(
    "SAM3_VisualizeAllMasks_Tool",
    {
        "image_path": image_path,
        "predictions": sam3_response["result"],
        "predictions_format": "sam3_server",
    },
)
print(viz_all["result"]["image_base64"])

viz_single = client.execute(
    "SAM3_VisualizeSingleMask_Tool",
    {
        "image_path": image_path,
        "predictions": sam3_response["result"],
        "predictions_format": "sam3_server",
        "mask_index": 0,
    },
)
print(viz_single["result"]["overlay_image_base64"])
print(viz_single["result"]["zoomed_image_base64"])
```
If you used `image_base64` for segmentation, pass the same `image_base64` to the visualize tools
instead of `image_path` (provide exactly one of the two).

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
