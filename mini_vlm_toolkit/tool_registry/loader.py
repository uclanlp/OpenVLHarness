import importlib
import inspect
import os
import sys

from mini_vlm_toolkit.tools.base import BaseTool

def _discover_tool_modules():
    tools_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")
    modules = []
    if not os.path.isdir(tools_dir):
        return modules
    for entry in sorted(os.listdir(tools_dir)):
        tool_path = os.path.join(tools_dir, entry, "tool.py")
        if os.path.isfile(tool_path):
            modules.append(f"mini_vlm_toolkit.tools.{entry}.tool")
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


def load_tools():
    tool_instances = {}
    tool_metadata = {}

    module_paths, auto_mode = _get_tool_modules()

    for module_path in module_paths:
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:
            if auto_mode:
                print(
                    f"Skipping tool module {module_path}: import failed: {exc}",
                    file=sys.stderr,
                )
                continue
            raise
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseTool) or obj is BaseTool:
                continue
            try:
                instance = obj()
            except Exception as exc:
                if auto_mode:
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
                if auto_mode:
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
            tool_instances[instance.tool_name] = instance
            tool_metadata[instance.tool_name] = instance.get_metadata()

    if not tool_instances:
        raise RuntimeError("No tools loaded")

    return tool_instances, tool_metadata
