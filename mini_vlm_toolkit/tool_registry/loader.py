import importlib
import inspect

from mini_vlm_toolkit.tools.base import BaseTool

TOOL_MODULES = [
    "mini_vlm_toolkit.tools.python_executor.tool",
    "mini_vlm_toolkit.tools.base_generator.tool",
    "mini_vlm_toolkit.tools.google_search.tool",
    "mini_vlm_toolkit.tools.web_search.tool",
    "mini_vlm_toolkit.tools.wikipedia_search.tool",
    "mini_vlm_toolkit.tools.python_coder.tool",
]


def load_tools():
    tool_instances = {}
    tool_metadata = {}

    for module_path in TOOL_MODULES:
        module = importlib.import_module(module_path)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseTool) or obj is BaseTool:
                continue
            instance = obj()
            tool_instances[instance.tool_name] = instance
            tool_metadata[instance.tool_name] = instance.get_metadata()

    if not tool_instances:
        raise RuntimeError("No tools loaded")

    return tool_instances, tool_metadata
