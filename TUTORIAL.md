# Tool Server + Client Tutorial

This tutorial is for colleagues training vision-language models to generate tool calls. It focuses on endpoint layout and client usage (not server/client internals).

## Prereqs

Install dependencies and start the server (install torch first). You can ask Di to start a server and forward to your machine as well.
```bash
pip install -r requirements-server.txt
export PYTHONPATH="$(pwd)"
./scripts/run_server.sh
```

By default the server listens at `http://127.0.0.1:8101`.

If you don't need to start the server on your end, you can install the lighter `requirements-client.txt`.

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
from mini_vlm_toolkit.client.client import ToolClient

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

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.client.parser import parse_tool_execute_args
from mini_vlm_toolkit.config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from mini_vlm_toolkit.llm.engine import LLMEngine

client = ToolClient("http://127.0.0.1:8101")
metadata = client.list_tools()["Python_Executor_Tool"]

prompt = (
    "Generate a single Python code block that calls this tool.\n"
    "The code must include exactly one line: execution = tool.execute(...).\n"
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
args = parse_tool_execute_args(llm_output)

response = client.execute("Python_Executor_Tool", args)
print(response["result"])
```

## Example 2: ZoomInImageByBBox tool (vision input)

`ZoomInImageByBBox_Tool` crops an image using a normalized bbox and optional padding. It returns a base64-encoded cropped image.

### Client-only call
```python
from mini_vlm_toolkit.client.client import ToolClient

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

### Call from an LLM (generate tool call, parse, execute)
```python
import os

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.client.parser import parse_tool_execute_args
from mini_vlm_toolkit.config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from mini_vlm_toolkit.llm.engine import LLMEngine

client = ToolClient("http://127.0.0.1:8101")
metadata = client.list_tools()["ZoomInImageByBBox_Tool"]
image_path = "tests/data/vision_tools_demo.jpg"

prompt = (
    "Generate a single Python code block that calls this tool.\n"
    "The code must include exactly one line: execution = tool.execute(...).\n"
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
args = parse_tool_execute_args(llm_output)

# Guardrails if the LLM omitted required fields.
args.setdefault("image_path", image_path)
args.setdefault("bbox", [0.05, 0.05, 0.45, 0.45])
args.setdefault("padding", 0.05)

response = client.execute("ZoomInImageByBBox_Tool", args)
print(response["result"])
```

## Example 3: SAM3 segment tool (vision input)

`SAM3_Segment_Tool` calls an external SAM3 server. Make sure the SAM3 server is running and reachable via `SAM3_SERVER_URL` (default `http://127.0.0.1:8083`).

### Client-only call
```python
from mini_vlm_toolkit.client.client import ToolClient

client = ToolClient("http://127.0.0.1:8101")
image_path = "tests/data/sam3_demo.jpg"

response = client.execute(
    "SAM3_Segment_Tool",
    {
        "prompt": "cow",
        "image_path": image_path,
        "max_masks": 3,
        "return_masks_png_b64": True,
    },
)
print(response["result"])
```

### Call from an LLM (generate tool call, parse, execute)
```python
import os

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.client.parser import parse_tool_execute_args
from mini_vlm_toolkit.config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from mini_vlm_toolkit.llm.engine import LLMEngine

client = ToolClient("http://127.0.0.1:8101")
metadata = client.list_tools()["SAM3_Segment_Tool"]
image_path = "tests/data/sam3_demo.jpg"

prompt = (
    "Generate a single Python code block that calls this tool.\n"
    "The code must include exactly one line: execution = tool.execute(...).\n"
    "Use keyword arguments only.\n"
    f"Tool name: SAM3_Segment_Tool\n"
    f"Tool metadata: {metadata}\n"
    f"Image path to use: {image_path}\n"
    "User request: Segment the cow in the image and return up to 3 masks.\n"
    "Return only the code block."
)

engine = LLMEngine(
    model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
    base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
    api_key=os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY),
)
llm_output = engine.generate(prompt, max_tokens=1024, temperature=0.4)
args = parse_tool_execute_args(llm_output)

# Guardrails if the LLM omitted required fields.
args.setdefault("prompt", "cow")
args.setdefault("image_path", image_path)
args.setdefault("max_masks", 3)
args.setdefault("return_masks_png_b64", True)

response = client.execute("SAM3_Segment_Tool", args)
print(response["result"])
```

## Notes for VLM tool-call training

- The server only sees the `tool_name` and `args`. If your model needs image context, handle that on the model side and pass the image path through the tool call.
- The parser expects a `tool.execute(...)` call in Python code. Keep the prompt strict to make parsing reliable.
- You can use `client.list_tools()` to capture tool metadata for training prompts or schema conditioning.
