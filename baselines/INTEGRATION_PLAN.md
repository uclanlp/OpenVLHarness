# Integration Plan: External Tool-Use Baselines (AdaReasoner, LATTE, REVPT)

## Context

We have three external tool-use MLLM baselines (AdaReasoner, LATTE, REVPT) that have been reproduced. Each has its own model, tool environment, and dependencies. We want to evaluate all three on our full benchmark suite using our existing eval framework, with minimal changes to the eval codebase and maximum isolation between baselines.

**Why HTTP endpoints (not Python imports):** The three baselines have incompatible dependency trees (AutoGen for LATTE, veRL for REVPT, different torch versions). They cannot share a Python environment. Each baseline also has its own tool servers (GroundingDINO, OCR, depth estimation, etc.) that need separate GPU/process management.

**Why NOT full OpenAI-compatible endpoints:** All three baselines run internal multi-turn tool loops. The OpenAI `/v1/chat/completions` spec is single-turn — making each baseline faithfully implement it (streaming, tool_calls arrays, etc.) is unnecessary work. The right abstraction is higher: each baseline exposes a single "generate" endpoint that encapsulates the full tool-use chain and returns the final answer.

---

## Architecture

```
Eval Framework (sandbox/)
  run_benchmark.py
    → build_agent(config) → ExternalBaselineAgent
      → HTTP POST {baseline_url}/generate
        → Returns {"prediction": ..., "trajectory": [...]}

                    ┌──────────────┐
                    │ AdaReasoner  │ :5001 (own conda + tool workers)
                    ├──────────────┤
                    │ LATTE        │ :5002 (own conda + inline tools)
                    ├──────────────┤
                    │ REVPT        │ :5003 (own conda + FastAPI tools)
                    └──────────────┘
```

---

## Part 1: Changes to the Eval Framework

### 1A. New agent class: `ExternalBaselineAgent`

**New file:** `mini_vlm_toolkit/agent/external_baseline.py`

A thin agent that:
- Does NOT extend `BaseAgent` (no `/v1/chat/completions` call, no `/v1/models` health check)
- Takes `baseline_url` (e.g., `http://localhost:5001`) and `model` (string for output directory naming — `run_benchmark.py:134` reads `agent.model`)
- `generate(message, **kwargs)`:
  1. Extracts text and image paths from the framework's message format (`[{type:"image", value:"/path"}, {type:"text", value:"question"}]`)
  2. Reads image files and base64-encodes them (baseline server may not share the filesystem)
  3. POSTs JSON to `{baseline_url}/generate`:
     ```json
     {"question": "...", "images": ["base64..."], "dataset": "...", "task_id": "..."}
     ```
  4. Parses response and returns `{"prediction": str, "extra_records": {"trajectory": ...}}`
- Health check via `GET {baseline_url}/health` in constructor (with retry/warning, not crash)
- Constructor signature:
  ```python
  def __init__(self, baseline_url, model="external", timeout=600, retry=3, verbose=False, **kwargs)
  ```

### 1B. Register the agent

**Modify:** `mini_vlm_toolkit/agent/__init__.py`

Follow the existing `DVDAgent` pattern:
```python
try:
    from .external_baseline import ExternalBaselineAgent
except Exception:
    ExternalBaselineAgent = None

if ExternalBaselineAgent is not None:
    AVAILABLE_AGENTS['ExternalBaselineAgent'] = ExternalBaselineAgent
```

### 1C. Config files

**New files in** `configs/baselines/`:

`adareasoner.yaml`:
```yaml
agent_name: ExternalBaselineAgent
agent_config_version: adareasoner_v1
agent_kwargs:
  baseline_url: "http://localhost:5001"
  model: "AdaReasoner-Qwen2VL-7B"
  timeout: 600
  retry: 3
```

`latte.yaml` and `revpt.yaml` — same structure, different URL/port/model name.

### 1D. Baselines folder

**New directory:** `baselines/`

Contents:
- `README.md` — instructions for baseline implementer (endpoint contract, launch steps, testing)
- `endpoint_contract.md` — the formal `/generate` and `/health` endpoint spec with request/response examples
- `test_endpoint.py` — script that sends a synthetic request to a baseline URL and validates the response structure

---

## Part 2: The Endpoint Contract (what the baseline implementer builds)

Each baseline is wrapped in a FastAPI server (`eval_server.py` in its own repo) with two endpoints:

### `GET /health`
```json
→ {"status": "ok", "baseline": "adareasoner"}
```

### `POST /generate`

**Request:**
```json
{
  "question": "What color is the car?",
  "images": ["<base64-encoded-image>", ...],
  "dataset": "MMBench_DEV_EN",
  "task_id": "42"
}
```

**Response:**
```json
{
  "prediction": "The car is red.",
  "trajectory": [
    {"turn": 1, "role": "assistant", "content": "Let me detect the car...", "tool_calls": [...]},
    {"turn": 1, "role": "tool", "content": "Detected: car at [120,50,340,280]"},
    {"turn": 2, "role": "assistant", "content": "The car is red."}
  ]
}
```

`trajectory` is optional but recommended for analysis. `prediction` is required.

**Key rules:**
- The server handles ALL tool orchestration internally — the eval framework only sees the final answer
- The server must handle concurrent requests (the eval framework sends `num-proc` requests in parallel)
- The server applies its own system prompt / input formatting internally
- Images may be large (multi-megapixel base64); set appropriate request size limits
- Create fresh agent state per request (no cross-contamination between samples)

---

## Part 3: Baseline Server Implementations

The baseline implementer writes an `eval_server.py` in each baseline's repo. All three follow the same endpoint contract above.

### AdaReasoner (`../AdaReasoner/eval_server.py`)
1. Load model via existing `VllmModels` / `Qwen2VLModel`
2. Initialize `ToolManager` connected to running tool workers
3. On `/generate`: decode images → build conversation → run tool loop (reuse `BaseToolInferencer` logic) → return final answer
4. **Prerequisite**: tool workers must be running (existing AdaReasoner launch scripts)

### LATTE (`../LATTE/eval_server.py`)
1. Initialize `UserAgent`, `MultimodalConversableAgent`, `Executor`, action classes
2. On `/generate`: decode images → save to temp files → `format_query()` → `initiate_chat()` → extract final answer from conversation
3. **Prerequisite**: GroundingDINO/RAM servers running (existing `scripts/serve_*.py`); vLLM server for the VLM if using API mode

### REVPT (`../REVPT/eval_server.py`)
1. This is the easiest — REVPT already has a clean `APIAgent.chat_with_tools()` interface
2. On `/generate`: decode images to PIL → create `APIAgent` → call `chat_with_tools(prompt, images)` → return response
3. **Prerequisite**: vLLM server for REVPT model + tool FastAPI services (existing `tools/lanuch_tools.py`)

---

## Part 4: Division of Responsibilities

### Eval framework side:
- [ ] Write `ExternalBaselineAgent` class (~80 lines)
- [ ] Register in `__init__.py` (~5 lines)
- [ ] Create 3 config YAMLs
- [ ] Write `baselines/README.md` and `baselines/endpoint_contract.md`
- [ ] Write `baselines/test_endpoint.py`

### Baseline implementer:
- [ ] Write `eval_server.py` for AdaReasoner (~100-200 lines)
- [ ] Write `eval_server.py` for LATTE (~100-200 lines)
- [ ] Write `eval_server.py` for REVPT (~100-200 lines)
- [ ] Write/adapt launch scripts for each tool environment
- [ ] Test each server with `test_endpoint.py`
- [ ] Run evaluations: `python run_benchmark.py --config configs/baselines/<name>.yaml --data <datasets>`

**Recommended order** (easiest → hardest):
1. REVPT — already has a clean `APIAgent.chat_with_tools()` interface
2. AdaReasoner — need to extract single-sample inference path from batch infrastructure
3. LATTE — deeply coupled to AutoGen's `initiate_chat` flow, most wrapping needed

---

## Part 5: Files to Create/Modify

| Action | File | Description |
|--------|------|-------------|
| **Create** | `mini_vlm_toolkit/agent/external_baseline.py` | ExternalBaselineAgent class |
| **Modify** | `mini_vlm_toolkit/agent/__init__.py` | Register new agent (3 lines) |
| **Create** | `configs/baselines/adareasoner.yaml` | Config for AdaReasoner |
| **Create** | `configs/baselines/latte.yaml` | Config for LATTE |
| **Create** | `configs/baselines/revpt.yaml` | Config for REVPT |
| **Create** | `baselines/README.md` | Baseline implementer documentation |
| **Create** | `baselines/endpoint_contract.md` | Formal endpoint spec |
| **Create** | `baselines/test_endpoint.py` | Smoke test script |

---

## Verification

1. Start a mock baseline server (simple FastAPI that returns a fixed answer) on localhost:5001
2. Run `python baselines/test_endpoint.py --url http://localhost:5001` — should pass
3. Run `python run_benchmark.py --config configs/baselines/adareasoner.yaml --data MMStar --num-examples 5` — should produce predictions and evaluation scores
4. Verify output directory structure: `outputs/AdaReasoner-Qwen2VL-7B/externalbaselineagent/...`
5. Verify trajectory data in `trajectories.jsonl`
