import copy
import importlib
import inspect
import os
import re
import sys

from tools.base import BaseTool


def _discover_tool_modules():
    tools_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")
    modules = []
    if not os.path.isdir(tools_dir):
        return modules
    for entry in sorted(os.listdir(tools_dir)):
        tool_path = os.path.join(tools_dir, entry, "tool.py")
        if os.path.isfile(tool_path):
            modules.append(f"tools.{entry}.tool")
    return modules


def _get_tool_modules():
    raw_modules = os.getenv("TOOL_MODULES")
    if raw_modules is None or not raw_modules.strip():
        return _discover_tool_modules(), True

    modules = [item.strip() for item in raw_modules.split(",") if item.strip()]
    if not modules:
        raise RuntimeError("TOOL_MODULES was set but empty")

    lowered = [item.lower() for item in modules]
    if "auto" in lowered:
        if len(modules) != 1:
            raise RuntimeError(
                "TOOL_MODULES must be either 'auto' or a comma-separated list of modules"
            )
        return _discover_tool_modules(), True

    return modules, False


def _canonical_to_snake_alias(tool_name):
    name = str(tool_name or "").strip()
    if name.endswith("_Tool"):
        name = name[: -len("_Tool")]
    name = name.replace("-", "_")
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"_+", "_", name)
    return name.lower().strip("_")


def _normalize_declared_aliases(tool_name, aliases):
    if aliases is None:
        return []
    if not isinstance(aliases, (list, tuple)):
        raise RuntimeError(f"{tool_name}: tool_aliases must be a list[str]")

    normalized = []
    seen = set()
    for alias in aliases:
        if not isinstance(alias, str):
            raise RuntimeError(f"{tool_name}: alias must be str, got {type(alias).__name__}")
        candidate = alias.strip()
        if not candidate:
            raise RuntimeError(f"{tool_name}: alias must be non-empty")
        if not candidate.isidentifier():
            raise RuntimeError(
                f"{tool_name}: alias '{candidate}' must be a valid Python identifier"
            )
        if candidate == tool_name:
            raise RuntimeError(
                f"{tool_name}: alias '{candidate}' cannot match the canonical tool name"
            )
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def _build_aliases_for_tool(tool_name, metadata):
    declared_aliases = _normalize_declared_aliases(
        tool_name, metadata.get("tool_aliases")
    )
    auto_alias = _canonical_to_snake_alias(tool_name)
    aliases = list(declared_aliases)
    if auto_alias and auto_alias != tool_name and auto_alias not in aliases:
        if not auto_alias.isidentifier():
            raise RuntimeError(
                f"{tool_name}: generated snake_case alias '{auto_alias}' is invalid"
            )
        aliases.append(auto_alias)
    return aliases


def _register_aliases(canonical_instances, canonical_metadata):
    tool_instances = {}
    tool_metadata = {}

    for tool_name, instance in canonical_instances.items():
        tool_instances[tool_name] = instance
        tool_metadata[tool_name] = canonical_metadata[tool_name]

    for tool_name, metadata in canonical_metadata.items():
        aliases = metadata.get("tool_aliases") or []
        for alias in aliases:
            if alias in canonical_instances:
                raise RuntimeError(
                    f"Alias collision: '{alias}' for '{tool_name}' conflicts with "
                    f"canonical tool '{alias}'"
                )
            if alias in tool_instances:
                existing = tool_metadata[alias].get("canonical_tool_name", alias)
                raise RuntimeError(
                    f"Alias collision: '{alias}' for '{tool_name}' conflicts with "
                    f"alias for canonical tool '{existing}'"
                )
            tool_instances[alias] = canonical_instances[tool_name]
            alias_metadata = copy.deepcopy(metadata)
            alias_metadata["tool_name"] = alias
            alias_metadata["is_alias"] = True
            alias_metadata["canonical_tool_name"] = tool_name
            tool_metadata[alias] = alias_metadata

    return tool_instances, tool_metadata


def load_tools():
    canonical_instances = {}
    canonical_metadata = {}

    module_paths, auto_mode = _get_tool_modules()

    for module_path in module_paths:
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:
            # Some modules intentionally opt into fail-fast behavior even when
            # TOOL_MODULES=auto, so configuration problems are never hidden.
            strict_import = module_path == "tools.subagent.tool"
            if auto_mode and not strict_import:
                print(
                    f"Skipping tool module {module_path}: import failed: {exc}",
                    file=sys.stderr,
                )
                continue
            raise
        strict_module = bool(getattr(module, "STRICT_TOOL_LOADING", False))
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseTool) or obj is BaseTool:
                continue
            try:
                instance = obj()
            except Exception as exc:
                if auto_mode and not strict_module:
                    print(
                        f"Skipping tool class {module_path}.{name}: init failed: {exc}",
                        file=sys.stderr,
                    )
                    continue
                raise
            issues = []
            try:
                issues = instance.sanity_check() or []
            except Exception as exc:
                issues = [str(exc)]
            if isinstance(issues, str):
                issues = [issues]
            if issues:
                formatted = "\n".join(f"- {issue}" for issue in issues)
                if auto_mode and not strict_module:
                    print(
                        "Skipping tool {name}: sanity check failed:\n{issues}".format(
                            name=instance.tool_name, issues=formatted
                        ),
                        file=sys.stderr,
                    )
                    continue
                raise RuntimeError(
                    f"Sanity check failed for {instance.tool_name}:\n{formatted}"
                )
            tool_name = instance.tool_name
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise RuntimeError("Tool name must be a non-empty string")
            if tool_name in canonical_instances:
                raise RuntimeError(f"Duplicate canonical tool name: {tool_name}")

            metadata = instance.get_metadata()
            tool_aliases = _build_aliases_for_tool(tool_name, metadata)
            metadata["tool_aliases"] = tool_aliases
            metadata["is_alias"] = False
            metadata["canonical_tool_name"] = tool_name
            metadata["tool_module_path"] = module_path
            if module_path.endswith(".tool"):
                processor_module = f"{module_path.rsplit('.', 1)[0]}.process_tool_output"
            else:
                processor_module = f"{module_path}.process_tool_output"
            metadata["tool_output_processor_module"] = processor_module

            canonical_instances[tool_name] = instance
            canonical_metadata[tool_name] = metadata

    if not canonical_instances:
        raise RuntimeError("No tools loaded")

    tool_instances, tool_metadata = _register_aliases(
        canonical_instances=canonical_instances,
        canonical_metadata=canonical_metadata,
    )

    return tool_instances, tool_metadata
