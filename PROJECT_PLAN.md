# Mini VLM Tool Kit

## Purpose
Build a minimal research toolkit that exposes the AgentFlow toolset via a small tool server and a simple client. The client can call tools directly or via LLM-generated tool-call code. The project is intentionally small, clean, and easy for multiple researchers to work on in parallel.

## Scope
- Implement a small tool server that loads tools and exposes an HTTP API.
- Implement a small tool client that can:
  1) invoke tools directly,
  2) invoke tools through the server,
  3) invoke tools via an LLM that emits tool-call code.
- Provide test scripts for each AgentFlow tool (3 tests each).
- Provide a single shell script to configure and run the server.

Non-goals:
- Production-grade auth, billing, or multi-tenant isolation.
- Full AgentFlow solver/planner stack.
- Model Context Protocol (MCP) integration (explicitly deferred).

## Reference Source
When referencing AgentFlow code or behavior, only use:
- `mini_vlm_tool_kit/reference/AgentFlow`

Do not reference any external path outside `mini_vlm_tool_kit`.

## Supported Tools (AgentFlow parity)
These tools are the initial compatibility target:
- Generalist_Solution_Generator_Tool
  - Summary: general LLM responder
  - Input: `query: str`
  - Output: `str`
- Ground_Google_Search_Tool
  - Summary: grounded Google search w/ optional citations
  - Input: `query: str`, `add_citations: bool`
  - Output: `str`
- Web_RAG_Search_Tool
  - Summary: RAG over a specific URL (fetch, chunk, embed, rank, summarize)
  - Input: `query: str`, `url: str`
  - Output: `str`
- Wikipedia_RAG_Search_Tool
  - Summary: Wikipedia search, LLM page selection, Web RAG per page
  - Input: `query: str`
  - Output: `dict`
- Python_Code_Generator_Tool
  - Summary: LLM generates Python snippet for math-like tasks, executes in restricted mode
  - Input: `query: str`
  - Output: `dict`

## Core Design
### Tool Interface
Each tool will follow the BaseTool-like contract:
- Metadata: `tool_name`, `tool_description`, `tool_version`, `input_types`, `output_type`, `demo_commands`, `user_metadata`.
- Required method: `execute(**kwargs)`.

### Tool Registry
- Load tool modules from `mini_vlm_tool_kit/tools/<tool_dir>/tool.py`.
- Map external tool names to internal module/class names (AgentFlow pattern).
- Store tool metadata and a cached instance per tool.

### Server
- Minimal FastAPI server with endpoints:
  - `GET /tools` -> list tool metadata
  - `POST /execute` -> run a tool by name and args
- The server executes tools in-process (no remote exec).
- Timeouts and basic input validation based on `input_types`.
- Optional simple log capture for debugging.

### Client
- A small client with three invocation paths:
  1) Direct local call (`tool.execute(...)`).
  2) Remote server call (`POST /execute`).
  3) LLM codegen call:
     - LLM generates code containing `execution = tool.execute(...)` lines.
     - Client parses these calls into `{tool_name, args}`.
     - Client calls the server with structured requests.

### LLM Engines
Allowed providers:
- OpenAI
- Together
- OpenAI-compatible vLLM endpoints

We will reuse a small engine wrapper (or a minimal subset from AgentFlow) but keep dependencies minimal. All LLM usage should be explicit and configurable via environment variables.

## Project Layout (planned)
```
mini_vlm_tool_kit/
  PROJECT_PLAN.md
  README.md
  tools/
    base.py
    base_generator/tool.py
    google_search/tool.py
    web_search/tool.py
    wikipedia_search/tool.py
    python_coder/tool.py
  tool_registry/
    loader.py
    mapping.py
  server/
    app.py
    models.py
  client/
    client.py
    parser.py
  llm/
    engine.py
  scripts/
    run_server.sh
  tests/
    test_base_generator.py
    test_google_search.py
    test_web_search.py
    test_wikipedia_search.py
    test_python_coder.py
  reference/
    AgentFlow/...
```

## Test Requirements (per tool)
Each test script should include 3 tests:
1) Direct invocation: instantiate tool and call `execute` locally.
2) Client invocation: call the tool via the HTTP server.
3) LLM invocation: prompt LLM to generate `execution = tool.execute(...)`, parse, then call server.

Each test should be runnable as a standalone script:
```
python tests/test_<tool>.py
```

## Server Script
`scripts/run_server.sh` should:
- set minimal env vars (if provided)
- start the server on a configurable host/port
- print the tool list endpoint for quick verification

Example usage:
```
./scripts/run_server.sh --host 127.0.0.1 --port 8001
```

## Environment Variables (planned)
- `OPENAI_API_KEY` for OpenAI tools and embeddings
- `TOGETHER_API_KEY` for Together
- `VLLM_BASE_URL` for OpenAI-compatible vLLM endpoints
- `DEFAULT_LLM_PROVIDER` optional, `openai|together|vllm`
- `DEFAULT_MODEL` optional model name

## Parallel Work Plan
- Person A: Tool registry + BaseTool + mapping logic.
- Person B: Server app and request/response models.
- Person C: Client + tool-call parser (AST-based, no exec).
- Person D: Test scripts per tool.

## Risks / Open Decisions
- Python tool execution safety: keep restricted and local.
- Web tools require network and API keys (Google, OpenAI). For tests, allow a small fallback mock or skip if env vars are missing.
- LLM codegen parsing: do not exec arbitrary code. Parse AST and only accept `tool.execute(...)` calls.

## Implementation Notes
- Keep code minimal, flat, and explicit.
- Avoid framework bloat or heavy dependencies beyond FastAPI, requests, and minimal LLM client.
- Keep all references to AgentFlow under `mini_vlm_tool_kit/reference/AgentFlow`.
