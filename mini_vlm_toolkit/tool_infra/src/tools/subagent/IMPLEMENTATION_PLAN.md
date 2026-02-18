# Subagent Tool Implementation Plan

## Goal
Expose a configured `BaseAgentWithTools` instance as a tool (`Subagent_Tool`) so a parent agent can call it with text/image inputs and receive structured results.

## Scope
- Implement in `mini_vlm_toolkit/tool_infra/src/tools/subagent`.
- Keep first version focused on one configured child agent.
- Preserve existing tool server `/execute` envelope contract.

## Proposed Tool Interface (MVP)
- `tool_name`: `Subagent_Tool`
- Inputs:
  - `message` (str, required): user text for the child agent.
  - `image_path` (str, optional): local image path.
  - `image_base64` (str, optional): base64/data URL image.
  - `max_iterations_override` (int, optional): bounded runtime override.
- Output (dict):
  - `answer` (str)
  - `ok` (bool)
  - `child_agent` (str)
  - `usage` (dict, optional)
  - `trajectory` (dict, optional, debug mode only)
  - `error` (dict, optional)

## Architecture
1. Create a child-agent factory in the tool module.
1. Load child config from env var path (e.g. `SUBAGENT_CONFIG_PATH`) using existing `build_agent`.
1. Validate child agent is `BaseAgentWithTools`.
1. Enforce child tool allowlist from child config (`tool_list`) and optional hard cap via env.
1. Convert tool inputs into agent input format:
   - text -> `[{"type": "text", "value": ...}]`
   - image -> include one image item using path or data URL
1. Execute `child_agent.generate(...)` and normalize return to tool output.

## Safety / Guardrails
- Recursion guard:
  - Disallow child `tool_list` containing `Subagent_Tool`.
  - Add runtime depth guard env (e.g. `SUBAGENT_MAX_DEPTH=1`) with thread-local depth counter.
- Timeouts:
  - Use child agent timeout; optional strict cap env (e.g. `SUBAGENT_TIMEOUT_SECONDS`).
- Payload limits:
  - Optional max input image bytes and max returned trajectory chars.
- Determinism:
  - Keep child default sampling params from config unless explicitly overridden.

## Dependency / Runtime Considerations
- Tool server process must import `mini_vlm_toolkit.agent` from repo root.
- Adjust launch/runtime docs so `PYTHONPATH` includes repo root in addition to tool infra `src`.
- Ensure `yaml` dependency is available in the runtime environment used for tool server.

## Test Plan
1. Unit tests (`tool_infra/tests/test_subagent_tool.py`):
   - Text-only happy path.
   - Image input via `image_base64`.
   - Invalid mixed image args (path + base64).
   - Missing config path / invalid config.
   - Recursion guard triggers when child tool list includes `Subagent_Tool`.
1. Contract tests:
   - `/jsonschema` exposes expected schema.
   - `/execute` returns envelope-compatible payload on success/failure.
1. Optional integration:
   - Parent agent with `Subagent_Tool` calls child that uses `Python_Executor_Tool`.

## Incremental Milestones
1. Skeleton tool + input schema + env sanity checks.
1. Child agent loading + cached instance lifecycle.
1. Input conversion + run + normalized output.
1. Recursion/depth guard.
1. Tests + docs (`README.md`, `TUTORIAL.md`).

## Open Decisions
- Should child trajectory be returned by default or only via debug flag?
- Do we support multiple named child agents in v1, or one fixed child per server process?
- Should parent pass system prompt overrides to child, or forbid for stability?
