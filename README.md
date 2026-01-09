# Mini VLM Tool Kit

Minimal research toolkit with a tool server and client. The first fully implemented tool is `Python_Executor_Tool`.

## Quick Start

Install dependencies:
```bash
pip install -r mini_vlm_tool_kit/requirements.txt
```

Start the server:
```bash
export PYTHONPATH=/home/<user>/vlm-colab/AgentFlow/mini_vlm_tool_kit
./mini_vlm_tool_kit/scripts/run_server.sh
```

List tools:
```bash
curl http://127.0.0.1:8001/tools
```

## CLI (no Python required)

Local execution (no server):
```bash
python -m mini_vlm_tool_kit.cli --tool Python_Executor_Tool --code "print(2+2)\nvalue=2+2"
```

Server execution:
```bash
python -m mini_vlm_tool_kit.cli --server http://127.0.0.1:8001 --tool Python_Executor_Tool --code "print(5*3)\nvalue=15"
```

## Tests

Each tool has a standalone test script. The Python executor test includes:
1) direct tool invocation
2) client invocation via server
3) LLM-generated tool call via client

Run:
```bash
export PYTHONPATH=/home/<user>/vlm-colab/AgentFlow/mini_vlm_tool_kit
python mini_vlm_tool_kit/tests/test_python_executor.py
```

LLM test environment:
- `LLM_MODEL` (required)
- `LLM_BASE_URL` (optional, default http://localhost:8003/v1)
- `LLM_API_KEY` (optional, default "local")

Example for vLLM Qwen3-32B at port 8003:
```bash
export LLM_MODEL=Qwen3-32B
export LLM_BASE_URL=http://localhost:8003/v1
```

## Notes
- The executor runs code with `exec()` and no sandboxing.
- Placeholder tools exist for AgentFlow parity and raise `NotImplementedError`.
