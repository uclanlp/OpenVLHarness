# Image Registry Walkthrough (BaseAgentWithTools)

This note explains how image references are tracked and surfaced to the model during the native tool loop in `BaseAgentWithTools`.

Scope:
- Runtime image registry built from user input images
- Environment Update user messages that report available image IDs
- `*_image_path` argument rewriting to `*_image_base64`
- Tool-output image extraction and re-registration for later rounds

## 1. Lifecycle at a Glance

Inside `BaseAgentWithTools._run(...)`, image flow is:

1. Build registry from current input images (`input_image_1`, `input_image_2`, ...).
2. Append an Environment Update user message after the initial user prompt.
3. For each assistant tool call, rewrite image path-like args from registry IDs/aliases to base64 data URLs.
4. Execute tool call via `/execute`.
5. Extract image blobs from tool result, keep `role=tool` content text-only, emit a separate multimodal `role=user` update message with those images, and register new refs (`tool_{tool_call_id}_image_<n>`). Default tool call IDs are generated as `call_<turn>_<idx>` (for example `call_1_1`), so image refs commonly look like `tool_call_1_1_image_1`.
6. After the tool-call turn, append another Environment Update user message (default emit policy is `always`).

Key methods in `mini_vlm_toolkit/agent/base.py`:
- `_build_image_registry_from_inputs`
- `_append_runtime_environment_messages`
- `_resolve_tool_call_args` / `_resolve_image_paths_in_args`
- `_build_tool_message_content`
- `_build_tool_image_user_message`
- `_register_tool_output_images`

Key hook class:
- `mini_vlm_toolkit/agent/runtime_environment.py::ImageEnvironment`

## 2. Example A: Single Image, Single Tool Call

User input:

```python
[
  {"type": "image", "value": "/data/demo.jpg"},
  {"type": "text", "value": "How many children are visible?"}
]
```

Registry (conceptually):

```json
{
  "input_image_1": {
    "source": "user_input",
    "aliases": [
      "input_image_1",
      "image_1",
      "/data/demo.jpg",
      "demo.jpg",
      "image",
      "image.jpg",
      "image.jpeg",
      "image.png"
    ]
  }
}
```

Environment Update message:

```text
Environment Update (not a new question):
Currently available image IDs for tool calls:
- input_image_1: user_input
When a tool needs image input, pass one of these IDs via the *_image_path field.
Do NOT invent image file names or paths.
```

If model emits:

```json
{"prompt": "children", "image_path": "image.jpg"}
```

Agent rewrites before `/execute`:

```json
{"prompt": "children", "image_base64": "data:image/..."}
```

Metadata recorded in tool response envelope:

```json
{
  "meta": {
    "image_resolution": {
      "rewrites": [
        {
          "field": "args.image_path",
          "target_field": "args.image_base64",
          "original": "image.jpg",
          "image_ref": "input_image_1"
        }
      ],
      "errors": []
    }
  }
}
```

## 3. Example B: Multi-Image + Multi-Round

User input:

```python
[
  {"type": "image", "value": "/data/left.png"},
  {"type": "image", "value": "/data/right.png"},
  {"type": "text", "value": "Compare the two images."}
]
```

Registry IDs:
- `input_image_1` (aliases include `left.png`)
- `input_image_2` (aliases include `right.png`)

If model emits:

```json
{
  "base_image_path": "left.png",
  "overlay_image_path": "right.png",
  "alpha": 0.6
}
```

Agent rewrites both fields:

```json
{
  "base_image_base64": "data:image/...",
  "overlay_image_base64": "data:image/...",
  "alpha": 0.6
}
```

Important behavior:
- Single-image fallback aliasing (`image.jpg` or unknown alias) is only used when exactly one image is registered.
- In multi-image mode, unresolved non-file aliases produce structured `unresolved_image_reference` errors (no guessing).

## 4. Example C: Tool Returns Images, Then Model Reuses Them

Suppose a tool returns:

```json
{
  "ok": true,
  "result": {
    "summary": "segmentation complete",
    "rendered_image_b64": "<base64...>",
    "pred_masks": ["<rle...>"]
  }
}
```

Agent behavior:

1. Builds text-only `role=tool` payload:

```json
{"ok": true, "result": {"summary": "segmentation complete", "pred_masks": ["<rle...>"]}, "error": null, "image_refs": ["tool_call_1_1_image_1"]}
```

2. Emits a separate multimodal `role=user` update carrying the tool images:

```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "Tool `SAM3_Segment_New_Tool` (tool_call_id=call_1_1) also returned the following image(s). Use these images for reasoning if helpful.\nSee the preceding role=tool message for structured output.\nImage refs: tool_call_1_1_image_1"
    },
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]
}
```

3. Registers reusable refs:
- `tool_call_1_1_image_1`
- Tool-name aliases are also registered (for example `SAM3_Segment_New_Tool_image_1`)

4. Appends a new runtime Environment Update message after the tool turn:

```text
Environment Update (not a new question):
Currently available image IDs for tool calls:
- input_image_1: user_input
- tool_call_1_1_image_1: tool_output:SAM3_Segment_New_Tool
When a tool needs image input, pass one of these IDs via the *_image_path field.
Do NOT invent image file names or paths.
```

5. Model can refer to them in later tool calls via `*_image_path`, for example:

```json
{"image_path": "tool_call_1_1_image_1", "prompt": "count children"}
```

Agent rewrites to `image_base64` automatically before execution.

## 5. Debugging / Verification Tips

- Run:

```bash
python examples/ad-hoc_ex_for_dev/base_agent_with_tools_image_response_example.py
```

- Enable verbose loop logs:

```bash
AGENT_DEBUG_TOOL_LOOP=1 python examples/ad-hoc_ex_for_dev/base_agent_with_tools_image_response_example.py
```

- Inspect:
  - `agent.last_run_messages` for exact message payloads sent to model
  - `agent.last_chat_history` for per-turn normalized tool responses and image rewrite metadata

## 6. Edge Cases to Keep in Mind

- If `image_path` is a real local filesystem path (`os.path.isfile(path)` is true), agent does not force registry rewrite.
- Registry deduplicates identical data URLs; multiple aliases can resolve to the same image ref.
- If there is exactly one registered image and model provides an unknown non-file alias, agent may fall back to that single image with metadata flag `single_image_registry_default`.
