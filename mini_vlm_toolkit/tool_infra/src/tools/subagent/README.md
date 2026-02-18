# Subagent Tools

This module exposes one tool per subagent config file.

## Enable
Set `SUBAGENT_CONFIG_DIR` to a directory containing YAML config files, then include `tools.subagent.tool` in tool loading.

## Config Schema
Each config must define:
- `subagent_name`: tool name, must end with `_Tool`
- `agent_name`: must be `BaseAgentWithTools`
- `agent_kwargs`: must include explicit `base`, `key`, `model`, `tool_base`, `timeout`, and non-empty `tool_list`

Optional fields:
- `subagent_aliases`: list of alias names
- `subagent_description`: tool description override

## Input
The generated subagent tools accept:
- `content`: single-turn list of content items
  - each item: `{"type": "text|image", "value": "..."}`

For image items, `value` may be a path, data URL, or raw base64.

## Output
Returns `{"ok", "answer", "subagent_name", "error"}`.
