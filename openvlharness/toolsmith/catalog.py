"""Load the generated tools of a catalog directory."""
import hashlib
import importlib.util
import inspect
import os
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


def fingerprint_files(files):
    """Return a SHA-256 over (file name, contents) pairs, in the order given."""
    digest = hashlib.sha256()
    for name, data in files:
        digest.update(os.fsencode(name) + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


def catalog_fingerprint(directory=None):
    """Return a SHA-256 over the file names and contents of the modules load_catalog imports.

    Raises:
        ValueError: The directory is missing.
    """
    return fingerprint_files((path.name, path.read_bytes()) for path in catalog_modules(directory))


def valid_metadata(tool):
    """Return whether a tool's name and description are strings and its input schema a mapping.

    tool is a GeneratedTool, or the mapping a worker reports for one.
    """
    field = tool.get if isinstance(tool, dict) else lambda name: getattr(tool, name, None)
    return (isinstance(field("tool_name"), str) and isinstance(field("tool_description"), str)
            and isinstance(field("input_schema"), dict))


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
            tool, a tool's name or description is not a string or its input schema
            not a mapping, or two tools share a name.
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
        if not valid_metadata(tool):
            raise ValueError(f"{path}: tool_name and tool_description must be strings and input_schema a mapping")
        if tool.tool_name in tools:
            raise ValueError(f"Duplicate generated tool name {tool.tool_name!r} in {path}")
        tool.client = client
        tools[tool.tool_name] = tool
    return tools
