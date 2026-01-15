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
- Port selected OpenThinkIMG tools into BaseTool implementations. Heavyweight models (SAM3, GroundingDINO, OCR, Depth) run as standalone model servers that the toolkit wraps, avoiding controller/worker infrastructure.

Non-goals:
- Production-grade auth, billing, or multi-tenant isolation.
- Full AgentFlow solver/planner stack.
- Model Context Protocol (MCP) integration (explicitly deferred).

## Reference Sources
When referencing external code or behavior, only use the repos under `reference/`:
- `reference/AgentFlow`
- `reference/OpenThinkIMG`
- `reference/VisualSketchpad`
- `reference/tool-vision`
- `reference/GroundingDINO`
- `reference/dpr-reproduction`

Do not reference any external path outside `mini-vlm-toolkit`.

## Supported Tools (AgentFlow parity)
These tools are the initial compatibility target:
- Gemini_Google_Search_Tool
  - Summary: Gemini-backed Google search w/ optional citations
  - Input: `query: str`, `add_citations: bool`
  - Output: `str`
- Serper_Google_Search_Tool
  - Summary: Serper-backed Google search that returns URLs
  - Input: `query: str`, `num_results: int`
  - Output: `list[str]`
- Single_URL_QA_Tool
  - Summary: QA over a specific URL (fetch, chunk, embed, rank, summarize)
  - Input: `query: str`, `url: str`
  - Output: `str`
- Python_Code_Generator_Tool
  - Summary: LLM generates Python snippet for math-like tasks, executes in restricted mode
  - Input: `query: str`
  - Output: `dict`

## Planned Tools (Vision tool ports)
We will port tool logic directly into `mini_vlm_toolkit/tools` and skip all external controller/worker or Gradio servers. Heavyweight models are served via dedicated FastAPI servers that the toolkit calls as wrappers.
Remaining unimplemented tools are deferred for now (not planned in the near term).
Planned tool: an LLM router that forwards OpenAI-style chat messages to vLLM servers (OpenAI-compatible), with env-based model->base_url mapping and one base URL per model.
Key decisions (recorded so new contributors do not need prior context):
- GroundingDINO: run a Hugging Face GroundingDINO model server (default `IDEA-Research/grounding-dino-base`) and wrap it from the toolkit; return data-first boxes/scores/labels (no VisualSketchpad wrapper).
- OCR: run an EasyOCR model server (CPU by default) and wrap it from the toolkit; EasyOCR downloads weights on first run.
- Depth: run a Depth-Anything model server (CPU default) and wrap it from the toolkit; server returns a depth colormap image.
- Crop: normalized coordinates are the primary interface (better composition with DINO outputs); pixel coords are optional via an explicit flag/enum.
- Zoom tools: keep both variants as separate tools.
  - `ZoomInSubfigure`: OpenThinkIMG behavior (subplot selection by description).
  - `ZoomInImageByBBox`: VisualSketchpad behavior (bbox-based crop with padding).
- Segmentation tools: keep both variants as separate tools.
  - `SegmentRegionAroundPoint`: point-guided segmentation implemented via SAM3 (replace OpenThinkIMG SAM2).
  - `SegmentAndMark`: full-image segmentation with numbered masks (VisualSketchpad Semantic-SAM).
- Visualization tools: SAM3 mask overlays for all masks and zoomed single-mask views (grounding-agent compatible).
  - SAM3 segment responses should optionally include a default all-masks visualization for quick review.
  - Tests: `tests/test_sam3_visualize.py` (SAM3 server) and `tests/test_sam3_visualize_synthetic.py` (synthetic).

Remaining OpenThinkIMG-derived tools (deferred):
- SegmentRegionAroundPoint: image + point -> localized mask (SAM3 model server)

Remaining VisualSketchpad-derived tools (deferred):
- SegmentAndMark: image + granularity -> annotated image + list of bboxes
- SlidingWindowDetection: image + objects -> candidate crops + boxes (uses detection internally)

## Core Design
### Tool Interface
Each tool will follow the BaseTool-like contract:
- Metadata: `tool_name`, `tool_description`, `tool_version`, `input_types`, `output_type`, `demo_commands`, `user_metadata`.
- Required method: `execute(**kwargs)`.

### Tool Registry
- Auto-discover tool modules from `mini_vlm_toolkit/tools/<tool_dir>/tool.py` when `TOOL_MODULES=auto` (or unset); otherwise load the explicit module list from `TOOL_MODULES`. Auto discovery skips tools that fail imports or sanity checks.
- Map external tool names to internal module/class names (AgentFlow pattern).
- Store tool metadata and a cached instance per tool.

### Server
- Minimal FastAPI server with endpoints:
  - `GET /tools` -> list tool metadata
  - `POST /execute` -> run a tool by name and args
- The server executes tools in-process; wrapper tools forward to local model servers for heavyweight models.
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
     - Parser should also accept JSON tool-call outputs that include a tool name and args.

### LLM Engines
Allowed providers:
- OpenAI
- Together
- OpenAI-compatible vLLM endpoints

We will reuse a small engine wrapper (or a minimal subset from AgentFlow) but keep dependencies minimal. All LLM usage should be explicit and configurable via environment variables.

## Project Layout (planned)
```
mini-vlm-toolkit/
  PROJECT_PLAN.md
  README.md
  mini_vlm_toolkit/
    __init__.py
    cli.py
    tools/
      base.py
      gemini_google_search/tool.py
      serper_google_search/tool.py
      single_url_qa/tool.py
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
    test_gemini_google_search.py
    test_serper_google_search.py
    test_single_url_qa.py
    test_python_coder.py
  reference/
    AgentFlow/...
    OpenThinkIMG/...
    VisualSketchpad/...
    tool-vision/...
    GroundingDINO/...
    dpr-reproduction/...
```

## Test Requirements (per tool)
Each test script should include 3 tests:
1) Direct invocation: instantiate tool and call `execute` locally.
2) Client invocation: call the tool via the HTTP server.
3) LLM invocation: prompt LLM to generate `execution = tool.execute(...)`, parse, then call server.

LLM tests should set `max_tokens` to 8192 to reduce the risk of truncated tool-call code.

Each test should be runnable as a standalone script:
```
python tests/test_<tool>.py
```

## Server Script
`scripts/run_server.sh` should:
- source the repo-root `.env` for main server settings
- start the server on a configurable host/port
- print the tool list endpoint for quick verification
It requires `LIGHT_TOOL_MODULES` (`auto` or a comma-separated list) or `HEAVY_TOOL_MODULES`, and appends `HEAVY_TOOL_MODULES` only when `LIGHT_TOOL_MODULES` is explicit.
Shell exports override `.env` values for the same keys.

Example usage:
```
cp .env.example .env
./scripts/run_server.sh
```

## Environment Variables (planned)
- `OPENAI_API_KEY` for OpenAI tools and embeddings
- `TOGETHER_API_KEY` for Together
- `VLLM_BASE_URL` for OpenAI-compatible vLLM endpoints
- `DEFAULT_LLM_PROVIDER` optional, `openai|together|vllm`
- `DEFAULT_MODEL` optional model name
Model server launch scripts live in their tool folders; they continue to use
tool-specific env vars and do not source `.env`.

## Parallel Work Plan
- Person A: Tool registry + BaseTool + mapping logic.
- Person B: Server app and request/response models.
- Person C: Client + tool-call parser (AST-based, no exec).
- Person D: Test scripts per tool.

## Risks / Open Decisions
- Python tool execution safety: keep restricted and local.
- Web tools require network and API keys (Google, OpenAI). For tests, allow a small fallback mock or skip if env vars are missing.
- LLM codegen parsing: do not exec arbitrary code. Parse AST and only accept `tool.execute(...)` calls.
Planned: clarify heavy tool dependencies vs main server requirements in docs and env setup.
Planned: refresh `TUTORIAL.md` or add a new tutorial after config/script changes.
Planned: decide whether LLM-based argument correction lives as a separate parser layer or an optional fallback inside the existing parser.
Planned: split parser into modules for (1) single-line calls, (2) JSON calls, (3) LLM-assisted fixer.
Planned: add a scaffolding helper for new tools.

## Implementation Notes
- Keep code minimal, flat, and explicit.
- Avoid framework bloat or heavy dependencies beyond FastAPI, requests, and minimal LLM client.
- Keep all reference materials under `reference/` (read-only).
