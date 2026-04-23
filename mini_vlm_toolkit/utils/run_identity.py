"""Run identity and manifest helpers for benchmark output versioning."""

from __future__ import annotations

import copy
import hashlib
import json
import os.path as osp
import re
from datetime import datetime, timezone
from typing import Any

import yaml


SENSITIVE_VALUE_PLACEHOLDER = "<REDACTED>"
_SENSITIVE_KEY_EXACT = {
    "key",
    "api_key",
    "token",
    "secret",
    "password",
    "authorization",
}
_SENSITIVE_KEY_SUBSTR = ("api_key", "secret", "password")
_SENSITIVE_KEY_SUFFIX = ("_key", "_token", "_secret", "_password", "_auth")


def parse_yaml_mapping(config_path: str, *, label: str) -> dict:
    """Load a YAML file and require a top-level mapping."""
    if not config_path or not osp.exists(config_path):
        raise FileNotFoundError(
            f"{label} file not found: {config_path}. "
            "Please provide a valid YAML file path."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"{label} file must contain a top-level mapping, got {type(data).__name__}."
        )
    return data


def require_config_version(config: dict, *, field_name: str, label: str) -> str:
    """Require a non-empty string config version field."""
    value = config.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{label} must define a non-empty top-level field '{field_name}'."
        )
    return value.strip()


def require_agent_model_name(agent_config: dict) -> str:
    """Require model name from agent config for output path layout."""
    agent_kwargs = agent_config.get("agent_kwargs")
    if not isinstance(agent_kwargs, dict):
        raise ValueError("Agent config must contain mapping field 'agent_kwargs'.")
    model = agent_kwargs.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Agent config 'agent_kwargs.model' must be a non-empty string.")
    return model.strip()


def require_judge_model_name(judge_config: dict) -> str:
    """Require judge model name for resume identity matching."""
    model = judge_config.get("model_name")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Judge config must define a non-empty top-level field 'model_name'.")
    return model.strip()


def sanitize_for_path(value: str, *, field_name: str) -> str:
    """Sanitize a user-provided identifier for safe path use."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty string.")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise ValueError(
            f"'{field_name}' becomes empty after sanitization. "
            "Use letters/numbers/._- characters."
        )
    return sanitized


def build_run_id(
    agent_config_version: str,
    judge_config_version: str,
    run_tag: str = "default",
) -> str:
    """Build a readable run id with explicit agent/judge prefixes and timestamp."""
    agent_part = sanitize_for_path(agent_config_version, field_name="agent_config_version")
    judge_part = sanitize_for_path(judge_config_version, field_name="judge_config_version")
    tag_part = sanitize_for_path(run_tag, field_name="run_tag")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"agent_{agent_part}_judge_{judge_part}_{timestamp}_{tag_part}"


def _is_sensitive_key(key: str) -> bool:
    key_lower = str(key).strip().lower()
    if key_lower in _SENSITIVE_KEY_EXACT:
        return True
    if key_lower.endswith(_SENSITIVE_KEY_SUFFIX):
        return True
    return any(part in key_lower for part in _SENSITIVE_KEY_SUBSTR)


def redact_sensitive_fields(value: Any) -> Any:
    """Return a deep-copied structure with sensitive values redacted."""
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                redacted[key] = SENSITIVE_VALUE_PLACEHOLDER
            else:
                redacted[key] = redact_sensitive_fields(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive_fields(item) for item in value]
    return copy.deepcopy(value)


def strip_sensitive_fields(value: Any) -> Any:
    """Return a deep-copied structure with sensitive fields removed."""
    if isinstance(value, dict):
        stripped = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                continue
            stripped[key] = strip_sensitive_fields(item)
        return stripped
    if isinstance(value, list):
        return [strip_sensitive_fields(item) for item in value]
    return copy.deepcopy(value)


def canonical_json(value: Any) -> str:
    """Deterministic JSON string for hash comparisons."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    """SHA256 hash helper."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_agent_match_hash(agent_config: dict) -> str:
    """Hash full agent config with sensitive fields removed."""
    payload = strip_sensitive_fields(agent_config)
    return sha256_text(canonical_json(payload))


def build_judge_model_match_hash(judge_model_name: str) -> str:
    """Hash judge identity payload (model_name only)."""
    payload = {"model_name": judge_model_name}
    return sha256_text(canonical_json(payload))


def build_run_manifest(
    *,
    run_id: str,
    agent_config_version: str,
    judge_config_version: str,
    agent_config: dict,
    judge_config: dict,
    config_path: str,
    judge_config_path: str,
) -> dict:
    """Build persisted run metadata for resume safety checks."""
    judge_model_name = require_judge_model_name(judge_config)
    return {
        "run_id": run_id,
        "agent_config_version": agent_config_version,
        "judge_config_version": judge_config_version,
        "agent_config_hash": build_agent_match_hash(agent_config),
        "judge_model_name": judge_model_name,
        "judge_model_hash": build_judge_model_match_hash(judge_model_name),
        "matching_policy": {
            "agent": "full_config_without_sensitive_fields",
            "judge": "model_name_only",
        },
        "config_paths": {
            "agent": osp.abspath(config_path),
            "judge": osp.abspath(judge_config_path),
        },
        "config_snapshot_redacted": {
            "agent": redact_sensitive_fields(agent_config),
            "judge": redact_sensitive_fields(judge_config),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest_compatibility(existing: dict, expected: dict) -> None:
    """Raise ValueError when an existing manifest conflicts with this run request."""
    mismatches = []
    if existing.get("agent_config_hash") != expected.get("agent_config_hash"):
        mismatches.append("agent config mismatch (excluding sensitive fields)")
    if existing.get("judge_model_name") != expected.get("judge_model_name"):
        mismatches.append("judge model mismatch")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "Run id collision detected: same run directory already exists but configs differ "
            f"({details}). Please bump 'agent_config_version' and/or "
            "'judge_config_version' and rerun."
        )

