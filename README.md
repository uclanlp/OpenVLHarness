# Mini VLM Tool Kit

Minimal research toolkit with a tool server and client. The first fully implemented tool is `Python_Executor_Tool`.

## Quick Start

Install dependencies:
```bash
pip install -r requirements-client.txt
```

Optional SAM3 server dependencies:
```bash
pip install -r requirements-server.txt
```

Start the server:
```bash
export PYTHONPATH="$(pwd)"
./scripts/run_server.sh
```

Start the server with a specific tool list:
```bash
export PYTHONPATH="$(pwd)"
export TOOL_MODULES="mini_vlm_toolkit.tools.python_executor.tool,mini_vlm_toolkit.tools.sam3_segment.tool"
./scripts/run_server.sh
```

List tools:
```bash
curl http://127.0.0.1:8101/tools
```

## CLI (no Python required)

Local execution (no server):
```bash
python -m mini_vlm_toolkit.cli --tool Python_Executor_Tool --code "print(2+2)\nvalue=2+2"
```

Server execution:
```bash
python -m mini_vlm_toolkit.cli --server http://127.0.0.1:8101 --tool Python_Executor_Tool --code "print(5*3)\nvalue=15"
```

## Tests

Each tool has a standalone test script. The Python executor test includes:
1) direct tool invocation
2) client invocation via server
3) LLM-generated tool call via client
   - The LLM-assisted step builds a prompt and parses tool args in the test script.
This direct + client + LLM-assisted pattern should be the default for new tool tests.

Run:
```bash
export PYTHONPATH="$(pwd)"
python tests/test_python_executor.py
```
Direct execution helper test:
```bash
export PYTHONPATH="$(pwd)"
python tests/test_execute_direct.py
```

LLM test environment:
- `LLM_MODEL` (optional, default Qwen/Qwen3-8B)
- `LLM_BASE_URL` (optional, default http://localhost:8004/v1)
- `LLM_API_KEY` (optional, default "local")

Example for vLLM Qwen3-8B at port 8004:
```bash
export LLM_MODEL=Qwen/Qwen3-8B
export LLM_BASE_URL=http://localhost:8004/v1
```

## TODO
- Implement remaining tools: Generalist_Solution_Generator_Tool.
- Add API key files for Gemini/Serper/OpenAI tools and run their tests when you return.
- Incorporate tools from the Visual Scratchpad paper.
- Incorporate tools from OpenThinking (https://arxiv.org/pdf/2505.08617).

## Notes
- The executor runs code with `exec()` and no sandboxing.
- Placeholder tools exist for AgentFlow parity and raise `NotImplementedError`.
- API key files are read from:
  - `mini_vlm_toolkit/tools/gemini_google_search/google_api_key`
  - `mini_vlm_toolkit/tools/serper_google_search/serper_api_key`
  - `mini_vlm_toolkit/tools/single_url_qa/openai_api_key`
- Reminder: run `python tests/test_gemini_google_search.py`, `python tests/test_serper_google_search.py`, and `python tests/test_single_url_qa.py`.
