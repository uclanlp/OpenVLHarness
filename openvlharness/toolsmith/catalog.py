"""Load the generated tools of a catalog directory."""
import hashlib
import importlib.util
import inspect
import logging
import os
from pathlib import Path

from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError

from .tool import GeneratedTool

logger = logging.getLogger(__name__)

DEFAULT_CATALOG = Path(__file__).with_name("tools")
# Schema keywords that make a validator load other documents, by URL or file path.
REFERENCE_KEYWORDS = {"$ref", "$id", "$schema", "$anchor", "$dynamicRef", "$dynamicAnchor", "$recursiveRef",
                      "$recursiveAnchor"}


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


def tool_problem(tool):
    """Return why the agent cannot offer a tool, or None.

    The name and description must be strings, and the input schema a valid Draft 7
    schema that refers to no other document (see REFERENCE_KEYWORDS). tool is a
    GeneratedTool, or the mapping a worker reports for one.
    """
    if not valid_metadata(tool):
        return "tool_name and tool_description must be strings and input_schema a mapping"
    schema = tool["input_schema"] if isinstance(tool, dict) else tool.input_schema
    try:
        found = sorted(REFERENCE_KEYWORDS & set(_keys(schema)))
        if found:
            return f"input_schema uses {', '.join(found)}, which would load other documents"
        Draft7Validator.check_schema(schema)
    except SchemaError as exc:
        return f"input_schema is not a valid JSON Schema: {exc.message}"
    except RecursionError:
        return "input_schema is nested too deeply"
    return None


def _keys(value):
    """Yield every mapping key in a JSON value."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def load_catalog(directory=None, *, client=None, strict=False):
    """Import every tool module in a catalog directory and instantiate its tool.

    A tool the agent cannot offer (see tool_problem) is skipped with a warning, so
    one bad module leaves the rest of the catalog usable.

    Args:
        directory: Directory holding one GeneratedTool subclass per .py file;
            files starting with an underscore are skipped. Defaults to the
            packaged catalog.
        client: Optional ToolClient the tools use to call other router tools.
        strict: Raise ValueError for a tool the agent cannot offer instead of
            skipping it, as Gate 1 does for a candidate.

    Returns:
        A mapping of tool names to tool instances, ordered by file name.

    Raises:
        ValueError: The directory is missing, a module does not define exactly one
            tool, two tools share a name, or strict is set and a tool cannot be offered.
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
        problem = tool_problem(tool)
        if problem and strict:
            raise ValueError(f"{path}: {problem}")
        if problem:
            logger.warning("ToolSmith catalog: skipping %s: %s", path.name, problem)
            continue
        if tool.tool_name in tools:
            raise ValueError(f"Duplicate generated tool name {tool.tool_name!r} in {path}")
        tool.client = client
        tools[tool.tool_name] = tool
    return tools
