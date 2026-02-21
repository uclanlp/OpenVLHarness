import copy
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.base import BaseTool

try:
    import yaml
except Exception:  # pragma: no cover - handled as configuration error.
    yaml = None

STRICT_TOOL_LOADING = True
SUBAGENT_CONFIG_DIR_ENV = "SUBAGENT_CONFIG_DIR"

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "array",
            "description": (
                "Single-turn child-agent input list. "
                "Each item is {'type': 'text|image', 'value': str}."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["text", "image"],
                    },
                    "value": {"type": "string"},
                },
                "required": ["type", "value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["content"],
    "additionalProperties": False,
}


def _repo_root() -> Path:
    # .../mini_vlm_toolkit/tool_infra/src/tools/subagent/tool.py -> repo root
    return Path(__file__).resolve().parents[5]


def _ensure_repo_root_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.append(root)


_AGENT_IMPORT_ERROR: Optional[str] = None
build_agent = None
BaseAgentWithTools = None

try:
    _ensure_repo_root_on_path()
    from mini_vlm_toolkit.agent import build_agent as _build_agent
    from mini_vlm_toolkit.agent.base import BaseAgentWithTools as _BaseAgentWithTools

    build_agent = _build_agent
    BaseAgentWithTools = _BaseAgentWithTools
except Exception as exc:  # pragma: no cover - resolved via sanity checks.
    _AGENT_IMPORT_ERROR = str(exc)


def _canonical_to_snake_alias(tool_name: str) -> str:
    name = str(tool_name or "").strip()
    if name.endswith("_Tool"):
        name = name[: -len("_Tool")]
    name = name.replace("-", "_")
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"_+", "_", name)
    return name.lower().strip("_")


def _is_valid_identifier(name: str) -> bool:
    return isinstance(name, str) and bool(name.strip()) and name.strip().isidentifier()


def _is_likely_base64(value: str) -> bool:
    if not isinstance(value, str):
        return False
    raw = value.strip()
    if len(raw) < 64:
        return False
    if raw.startswith("data:image/"):
        return True
    if raw.startswith("base64:"):
        raw = raw.split(":", 1)[1]
    return bool(re.fullmatch(r"[A-Za-z0-9+/=\s]+", raw))


def _normalize_image_value(value: str) -> str:
    raw = value.strip()
    if raw.startswith("data:image/"):
        return raw
    if raw.startswith("base64:"):
        raw = raw.split(":", 1)[1]
    if _is_likely_base64(raw):
        return f"data:image/png;base64,{raw}"
    return value


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fin:
        data = yaml.safe_load(fin)
    if not isinstance(data, dict):
        raise ValueError("config must parse to a dictionary")
    return data


def _validate_string_field(source: Dict[str, Any], key: str, *, required: bool = True) -> Optional[str]:
    value = source.get(key)
    if value is None:
        if required:
            raise ValueError(f"missing required field '{key}'")
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"field '{key}' must be a non-empty string")
    return value.strip()


def _validate_aliases(raw_aliases: Any) -> List[str]:
    if raw_aliases is None:
        return []
    if not isinstance(raw_aliases, list):
        raise ValueError("subagent_aliases must be a list[str]")
    aliases: List[str] = []
    seen = set()
    for item in raw_aliases:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("subagent_aliases entries must be non-empty strings")
        alias = item.strip()
        if not alias.isidentifier():
            raise ValueError(f"subagent_aliases entry '{alias}' must be a valid Python identifier")
        if alias in seen:
            continue
        seen.add(alias)
        aliases.append(alias)
    return aliases


def _load_subagent_spec(path: Path) -> Dict[str, Any]:
    cfg = _read_yaml(path)

    subagent_name = _validate_string_field(cfg, "subagent_name", required=True)
    if not subagent_name.endswith("_Tool"):
        raise ValueError("subagent_name must end with '_Tool'")
    if not _is_valid_identifier(subagent_name):
        raise ValueError("subagent_name must be a valid Python identifier")

    agent_name = _validate_string_field(cfg, "agent_name", required=True)
    if agent_name != "BaseAgentWithTools":
        raise ValueError("agent_name must be 'BaseAgentWithTools'")

    agent_kwargs = cfg.get("agent_kwargs")
    if not isinstance(agent_kwargs, dict):
        raise ValueError("agent_kwargs must be a dictionary")

    # Full child isolation: all connection/auth fields must be explicit.
    for key in ("base", "key", "model", "tool_base"):
        value = agent_kwargs.get(key)
        if not isinstance(value, str):
            raise ValueError(f"agent_kwargs.{key} must be an explicit string")
        if key != "key" and not value.strip():
            raise ValueError(f"agent_kwargs.{key} must be non-empty")

    timeout = agent_kwargs.get("timeout")
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("agent_kwargs.timeout must be a positive integer")

    tool_list = agent_kwargs.get("tool_list")
    if not isinstance(tool_list, list) or not tool_list:
        raise ValueError("agent_kwargs.tool_list must be a non-empty list[str] (null is not allowed)")
    for tool_name in tool_list:
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("agent_kwargs.tool_list entries must be non-empty strings")

    aliases = _validate_aliases(cfg.get("subagent_aliases"))

    description = cfg.get("subagent_description")
    if description is None:
        description = (
            f"Execute child agent '{subagent_name}' with its own fixed config and restricted tool set. "
            "Accepts chat-style messages with text/image content and returns the child's final answer."
        )
    elif not isinstance(description, str) or not description.strip():
        raise ValueError("subagent_description must be a non-empty string when provided")

    return {
        "config_path": str(path),
        "tool_name": subagent_name,
        "tool_aliases": aliases,
        "tool_description": description,
        "child_tool_list": [item.strip() for item in tool_list],
    }


def _discover_subagent_specs(config_dir: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    specs: List[Dict[str, Any]] = []

    if yaml is None:
        return [], ["PyYAML is required to load subagent configs."]

    raw = str(config_dir or "").strip()
    if not raw:
        return [], [f"{SUBAGENT_CONFIG_DIR_ENV} is empty."]

    cfg_dir = Path(raw)
    if not cfg_dir.is_dir():
        return [], [f"{SUBAGENT_CONFIG_DIR_ENV} must point to an existing directory: {cfg_dir}"]

    cfg_paths = sorted([*cfg_dir.glob("*.yaml"), *cfg_dir.glob("*.yml")])
    if not cfg_paths:
        return [], [f"No YAML configs found in {cfg_dir}"]

    seen_tool_names: Dict[str, str] = {}
    for cfg_path in cfg_paths:
        try:
            spec = _load_subagent_spec(cfg_path)
        except Exception as exc:
            errors.append(f"{cfg_path}: {exc}")
            continue

        tool_name = spec["tool_name"]
        existing = seen_tool_names.get(tool_name)
        if existing is not None:
            errors.append(
                f"Duplicate subagent_name '{tool_name}' in configs: {existing} and {cfg_path}"
            )
            continue

        seen_tool_names[tool_name] = str(cfg_path)
        specs.append(spec)

    if errors:
        return [], errors

    blocked_names_casefold = set()
    for spec in specs:
        canonical = spec["tool_name"]
        blocked_names_casefold.add(canonical.casefold())
        auto_alias = _canonical_to_snake_alias(canonical)
        if auto_alias:
            blocked_names_casefold.add(auto_alias.casefold())
        for alias in spec.get("tool_aliases") or []:
            blocked_names_casefold.add(alias.casefold())

    for spec in specs:
        for child_tool in spec["child_tool_list"]:
            if child_tool.casefold() in blocked_names_casefold:
                errors.append(
                    f"{spec['config_path']}: agent_kwargs.tool_list contains forbidden subagent tool '{child_tool}'"
                )

    if errors:
        return [], errors
    return specs, []


def _failure_payload(subagent_name: str, code: str, message: str, detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    error: Dict[str, Any] = {
        "code": code,
        "message": str(message),
    }
    if isinstance(detail, dict) and detail:
        error["detail"] = detail
    return {
        "ok": False,
        "answer": "",
        "subagent_name": subagent_name,
        "error": error,
    }


def _success_payload(subagent_name: str, answer: str) -> Dict[str, Any]:
    return {
        "ok": True,
        "answer": answer,
        "subagent_name": subagent_name,
        "error": None,
    }


def _normalize_content(content: Any) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    if not isinstance(content, list) or not content:
        return None, {
            "code": "invalid_content",
            "message": "content must be a non-empty list",
        }

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(content):
        if not isinstance(item, dict):
            return None, {
                "code": "invalid_content",
                "message": "each content item must be an object",
                "detail": {"index": idx, "type": type(item).__name__},
            }

        part_type = item.get("type")
        part_value = item.get("value")
        if part_type not in {"text", "image"}:
            return None, {
                "code": "invalid_content_item_type",
                "message": "content.type must be 'text' or 'image'",
                "detail": {
                    "content_index": idx,
                    "type": part_type,
                },
            }
        if not isinstance(part_value, str) or not part_value.strip():
            return None, {
                "code": "invalid_content_item_value",
                "message": "content.value must be a non-empty string",
                "detail": {
                    "content_index": idx,
                },
            }

        value_out = part_value
        if part_type == "image":
            value_out = _normalize_image_value(part_value)
        normalized.append({"type": part_type, "value": value_out})

    return normalized, None


def _safe_answer_from_prediction(prediction: Any) -> str:
    if prediction is None:
        return ""
    if isinstance(prediction, str):
        return prediction
    return str(prediction)


def _error_tool_class(errors: List[str]):
    issues = [str(item) for item in errors if str(item).strip()]

    def __init__(self):
        BaseTool.__init__(
            self,
            tool_name="Subagent_Config_Error_Tool",
            tool_description=(
                "Sentinel tool used when subagent config discovery fails. "
                "Server startup should fail under strict loading."
            ),
            tool_version="0.1.0",
            input_types={"content": "list[dict] - single-turn content items"},
            input_schema=copy.deepcopy(_INPUT_SCHEMA),
            output_type="dict - ok, answer, subagent_name, error",
        )
        self._issues = list(issues)

    def sanity_check(self):
        return list(self._issues)

    def execute(self, content):
        return _failure_payload(
            subagent_name="Subagent_Config_Error_Tool",
            code="subagent_configuration_error",
            message="Subagent tool configuration is invalid.",
            detail={"issues": list(self._issues)},
        )

    return type(
        "SubagentConfigErrorTool",
        (BaseTool,),
        {
            "__init__": __init__,
            "sanity_check": sanity_check,
            "execute": execute,
            "__module__": __name__,
        },
    )


def _make_subagent_tool_class(spec: Dict[str, Any]):
    tool_name = spec["tool_name"]
    config_path = spec["config_path"]
    tool_aliases = list(spec.get("tool_aliases") or [])
    tool_description = spec["tool_description"]

    class_name = "Subagent_" + re.sub(r"[^A-Za-z0-9_]", "_", tool_name)

    def __init__(self):
        BaseTool.__init__(
            self,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_version="0.1.0",
            input_types={
                "content": "list[dict] - single-turn content items [{type,value}]",
            },
            input_schema=copy.deepcopy(_INPUT_SCHEMA),
            tool_aliases=list(tool_aliases),
            output_type="dict - ok, answer, subagent_name, error",
            user_metadata={
                "subagent_config_path": config_path,
                "child_tool_list": list(spec.get("child_tool_list") or []),
            },
        )
        self._config_path = config_path
        self._subagent_name = tool_name
        self._agent = None
        self._agent_lock = threading.Lock()

    def sanity_check(self):
        issues = []
        if _AGENT_IMPORT_ERROR is not None:
            issues.append(f"Failed to import mini_vlm_toolkit.agent: {_AGENT_IMPORT_ERROR}")
        return issues

    def _get_or_build_agent(self):
        if self._agent is not None:
            return self._agent
        with self._agent_lock:
            if self._agent is not None:
                return self._agent
            agent = build_agent(self._config_path)
            if BaseAgentWithTools is None or not isinstance(agent, BaseAgentWithTools):
                raise TypeError(
                    f"Config {self._config_path} must build BaseAgentWithTools, got {type(agent).__name__}"
                )
            self._agent = agent
        return self._agent

    def execute(self, content):
        normalized_content, normalize_error = _normalize_content(content)
        if normalize_error is not None:
            return _failure_payload(
                subagent_name=self._subagent_name,
                code=normalize_error.get("code", "invalid_content"),
                message=normalize_error.get("message", "Invalid content payload."),
                detail=normalize_error.get("detail"),
            )

        try:
            agent = self._get_or_build_agent()
            result = agent.generate(normalized_content)
            answer = _safe_answer_from_prediction((result or {}).get("prediction"))
            return _success_payload(self._subagent_name, answer)
        except Exception as exc:
            return _failure_payload(
                subagent_name=self._subagent_name,
                code="subagent_execution_error",
                message=str(exc),
                detail={"exception_type": type(exc).__name__},
            )

    return type(
        class_name,
        (BaseTool,),
        {
            "__init__": __init__,
            "sanity_check": sanity_check,
            "_get_or_build_agent": _get_or_build_agent,
            "execute": execute,
            "__module__": __name__,
        },
    )


def _bootstrap_subagent_tools():
    config_dir = str(os.getenv(SUBAGENT_CONFIG_DIR_ENV, "")).strip()
    if not config_dir:
        return [], []

    if _AGENT_IMPORT_ERROR is not None:
        return [], [f"mini_vlm_toolkit.agent import failed: {_AGENT_IMPORT_ERROR}"]

    specs, errors = _discover_subagent_specs(config_dir)
    if errors:
        return [], errors
    return specs, []


SUBAGENT_TOOL_SPECS, SUBAGENT_DISCOVERY_ERRORS = _bootstrap_subagent_tools()
SUBAGENT_TOOL_CLASSES = []

if SUBAGENT_DISCOVERY_ERRORS:
    SUBAGENT_TOOL_CLASSES.append(_error_tool_class(SUBAGENT_DISCOVERY_ERRORS))
else:
    for _spec in SUBAGENT_TOOL_SPECS:
        SUBAGENT_TOOL_CLASSES.append(_make_subagent_tool_class(_spec))

for _cls in SUBAGENT_TOOL_CLASSES:
    globals()[_cls.__name__] = _cls
