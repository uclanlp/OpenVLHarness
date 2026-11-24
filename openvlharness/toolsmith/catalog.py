"""Load the generated tools of a catalog directory."""
import importlib.util
import inspect
from pathlib import Path

from .tool import GeneratedTool

DEFAULT_CATALOG = Path(__file__).with_name("tools")


def catalog_modules(directory=None):
    """Return the module files load_catalog imports from a catalog directory, in load order.

    Raises:
        ValueError: The directory is missing.
    """
    directory = Path(directory or DEFAULT_CATALOG)
    if not directory.is_dir():
        raise ValueError(f"ToolSmith catalog {str(directory)!r} is not a directory")
    return [path for path in sorted(directory.glob("*.py")) if not path.name.startswith("_")]


def load_catalog(directory=None, *, client=None):
    """Import every tool module in a catalog directory and instantiate its tool.

    Args:
        directory: Directory holding one GeneratedTool subclass per .py file;
            files starting with an underscore are skipped. Defaults to the
            packaged catalog.
        client: Optional ToolClient the tools use to call other router tools.

    Returns:
        A mapping of tool names to tool instances, ordered by file name.

    Raises:
        ValueError: The directory is missing, a module does not define exactly one
            tool, or two tools share a name.
    """
    tools = {}
    for path in catalog_modules(directory):
        spec = importlib.util.spec_from_file_location(f"toolsmith_catalog_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classes = [value for value in vars(module).values() if inspect.isclass(value)
                   and issubclass(value, GeneratedTool) and value.__module__ == module.__name__]
        if len(classes) != 1:
            raise ValueError(f"{path} must define exactly one GeneratedTool subclass")
        tool = classes[0]()
        if tool.tool_name in tools:
            raise ValueError(f"Duplicate generated tool name {tool.tool_name!r} in {path}")
        tool.client = client
        tools[tool.tool_name] = tool
    return tools
