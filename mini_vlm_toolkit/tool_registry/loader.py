import importlib
import inspect
import os

from mini_vlm_toolkit.tools.base import BaseTool

DEFAULT_TOOL_MODULES = [
    "mini_vlm_toolkit.tools.python_executor.tool",
    "mini_vlm_toolkit.tools.base_generator.tool",
    "mini_vlm_toolkit.tools.gemini_google_search.tool",
    "mini_vlm_toolkit.tools.serper_google_search.tool",
    "mini_vlm_toolkit.tools.single_url_qa.tool",
    "mini_vlm_toolkit.tools.python_coder.tool",
    "mini_vlm_toolkit.tools.sam3_segment.tool",
]


def _get_tool_modules():
    raw_modules = os.getenv("TOOL_MODULES")
    if raw_modules is None:
        return DEFAULT_TOOL_MODULES

    modules = [item.strip() for item in raw_modules.split(",") if item.strip()]
    if not modules:
        raise RuntimeError("TOOL_MODULES was set but empty")
    return modules


def load_tools():
    tool_instances = {}
    tool_metadata = {}

    for module_path in _get_tool_modules():
        module = importlib.import_module(module_path)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseTool) or obj is BaseTool:
                continue
            instance = obj()
            issues = []
            try:
                issues = instance.sanity_check() or []
            except Exception as exc:
                issues = [str(exc)]
            if isinstance(issues, str):
                issues = [issues]
            if issues:
                formatted = "\n".join(f"- {issue}" for issue in issues)
                raise RuntimeError(
                    f"Sanity check failed for {instance.tool_name}:\n{formatted}"
                )
            tool_instances[instance.tool_name] = instance
            tool_metadata[instance.tool_name] = instance.get_metadata()

    if not tool_instances:
        raise RuntimeError("No tools loaded")

    return tool_instances, tool_metadata
