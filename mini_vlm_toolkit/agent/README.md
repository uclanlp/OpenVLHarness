# Agent Module

## 1. Introduction
`mini_vlm_toolkit/agent` provides configurable agents for benchmark inference.

- Use `BaseAgent` for direct model calls without tool execution.
- Use `BaseAgentWithTools` for native OpenAI-style `tool_calls` with the tool server.
- Build from YAML configs via `build_agent` in `mini_vlm_toolkit/agent/__init__.py`.

This module is used by `run_benchmark.py` to run dataset inference and evaluation.

## 2. Walkthrough: BaseAgent vs BaseAgentWithTools
### `BaseAgent` (basic generation loop)
- Validates model server reachability (`/v1/models`) at init.
- Normalizes raw multimodal inputs (for example `text` / `image`) into chat payload format.
- Sends requests to `/v1/chat/completions`.
- Retries on failure and returns final text answer.

### `BaseAgentWithTools` (native tool loop)
- Inherits `BaseAgent`.
- In normal benchmark usage, it is configured from YAML (`agent_kwargs`) via `build_agent`.
- Loads tool metadata from tool server (`/tools`, `/jsonschema`).
- Calls model with `tools` and `tool_choice`.
- Reads native assistant `tool_calls`, enforces at most one call per assistant turn, and executes the single call via tool server `/execute`.
- Routes normalized tool payloads through `PersistentStateMemory` (default: `ImageStateMemory`) to produce model-facing messages.
- Appends structured tool outputs as text-only `role=tool` messages.
- If tool outputs include images, appends separate multimodal `role=user` messages (short text + `image_url` parts), then continues.
- Appends runtime Environment Update `role=user` messages and continues until final answer.
- Returns debug traces under `extra_records.trajectory` (`chat_history` + final model-fed `messages`).

## 3. Example Usage
1. Single-example debug loop (tool calling):
```bash
python examples/ad-hoc_ex_for_dev/base_agent_with_tools_chat_example_stateless.py
```

2. Single-example debug loop (image + tool response handling):
```bash
python examples/ad-hoc_ex_for_dev/base_agent_with_tools_image_response_example_stateless.py
```

3. Image registry lifecycle walkthrough (IDs, rewrites, and tool-output image refs):
```bash
cat examples/ad-hoc_ex_for_dev/IMAGE_REGISTRY_WALKTHROUGH.md
```

4. Endpoint capability probe for multimodal tool-response handling:
```bash
python examples/ad-hoc_ex_for_dev/multimodal_tool_visibility_probe.py
```

5. Dataset run through benchmark entrypoint:
```bash
python run_benchmark.py --config configs/base_agent_with_tools.yaml --data MMStar --num-proc 4
```

Notes:
- Example scripts are under `examples/ad-hoc_ex_for_dev/`.
- Agent configs are under `configs/`.
- Benchmark runs require top-level `agent_config_version` in agent config and `judge_config_version` in judge config.
- For richer tool-loop logs in the example scripts, set `AGENT_DEBUG_TOOL_LOOP=1`.
- For direct agent calls, pass `debug_tool_loop=True` to `generate(...)` / `_run(...)`.

## 4. Extending to a Dataset-Specific Agent
- Choose a base class: `BaseAgent` (no tools) or `BaseAgentWithTools` (tool execution).
- Register your class in `mini_vlm_toolkit/agent/__init__.py`.
- Add a YAML config in `configs/`.
- For unified VLM-agent-training efforts, prefer changing system prompt and tool set in config; avoid adding complex custom agent logic unless strictly necessary.

## 5. Notes
### Tool-calling workflow
- Assistant emits native `tool_calls`.
- Agent normalizes native tool calls, rejects multi-call assistant messages with tool-error feedback, and posts only a single call to tool server `/execute`.
- Tool payloads are normalized into `{ok, result, error, tool_name, tool_call_id, meta}` for internal history/tracing.
- `PersistentStateMemory` transforms each normalized payload into model-facing messages while preserving native tool protocol (`role=tool` + matching `tool_call_id`).
- The `role=tool` message content is text-only JSON (`ok/result/error` plus optional `image_refs`).
- If tool outputs include images, agent sends a separate multimodal `role=user` update message with those image parts.
- `chat_history` keeps raw tool payloads in `tool_responses` and compact memory emission traces in `memory_outputs` (with indices into `trajectory.messages`).
- Tool messages are fed back to the model until a non-tool final answer is returned.
- If the last allowed turn ends with tool calls, agent makes one final completion with `tool_choice="none"` to force a text answer.

### Image-related special handling
- Input images are registered as runtime IDs (for example `input_image_1`) and surfaced in Environment Update user messages.
- If a tool arg uses `*_image_path`, agent tries to resolve it to runtime image data and rewrites to `*_image_base64`.
- Unresolvable image references return structured `unresolved_image_reference` errors instead of silent failure.
- Base64 image outputs from tools are extracted, tracked as reusable image refs, and surfaced via a separate multimodal `role=user` update message.

### Key runtime controls
- `tool_choice`: `"auto"`, `"none"`, or forced function selection.
- `tool_list`: limit to a tool subset.
- `prompt_version`: `"v1"` (class default) or `"v2"` prompt templates. `configs/base_agent_with_tools.yaml` currently sets `"v2"`.
- `reasoning_effort`: optional GPT-5 reasoning knob (`"minimal"`, `"low"`, `"medium"`, `"high"`), applied on OpenAI endpoints.
- `persistent_state_memory`: single dict config, default `{"name": "image", "emit_policy": "always"}`.
- `persistent_state_memory` currently supports `name: "image"` and can be disabled with names `"none"`, `"off"`, or `"disabled"`.
- `emit_policy`: image memory emit behavior, `"always"` (default) or `"on_change"`.
- `max_iterations`: cap tool loop turns.
- `extra_body`: pass provider-specific fields to chat/completions.
- `debug_tool_loop`: print compact per-turn execution traces.

### Known pitfalls and guardrails
- Model endpoint must support native `tool_calls`; text-formatted pseudo-calls are not used.
- Tool server must expose `/tools`, `/jsonschema`, and `/execute`.
- Do not hardcode image filenames in prompts; use runtime image IDs from Environment Update messages.
- Check tool response `ok/error/meta` before trusting results.
