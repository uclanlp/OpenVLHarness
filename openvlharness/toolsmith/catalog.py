"""Load the generated tools of a catalog directory."""
import hashlib
import importlib.util
import inspect
import json
import logging
import os
from pathlib import Path

from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError

from .errors import CatalogLoadError, describe
from .tool import GeneratedTool

logger = logging.getLogger(__name__)

DEFAULT_CATALOG = Path(__file__).with_name("tools")
# The version of the rules load_catalog reads a catalog by: which modules it imports, and which
# tools it offers, skips or refuses (tool_problem and the keyword sets below). The catalog
# fingerprint includes it, so the same files read under other rules are another catalog to a run's
# identity and to the router check. Bump it whenever those rules change, stricter or looser.
LOADER_RULES = 2  # 2: the root input schema must be an object
# Schema keywords that refer to other documents or to other parts of the schema, or pick another
# draft. All are refused, local ones too: two local references per level make a small schema
# take exponential time to validate, and a $schema in a subschema switches to a draft whose
# keywords (extends, id, a schema as a type) the scan below does not model.
REFERENCE_KEYWORDS = {"$ref", "$id", "$schema", "$anchor", "$dynamicRef", "$dynamicAnchor", "$recursiveRef",
                      "$recursiveAnchor"}
# Schema keywords whose regular expressions can take exponential time to match an argument
# ("^([A-Za-z]+\\s?)+$" against a long title ending in "!"), with no time limit.
PATTERN_KEYWORDS = {"pattern", "patternProperties"}
# Generated-tool schemas may use neither: they can fetch documents or make argument validation slow.
REFUSED_KEYWORDS = REFERENCE_KEYWORDS | PATTERN_KEYWORDS
# Keywords whose value is a subschema, a list of subschemas, or a mapping of names to subschemas
# (Draft 7, and the later drafts' names). Elsewhere a mapping is data: property names, enum values.
SUBSCHEMA = {"additionalItems", "additionalProperties", "contains", "else", "if", "items", "not", "propertyNames",
             "then", "unevaluatedItems", "unevaluatedProperties", "contentSchema"}
SUBSCHEMA_LISTS = {"allOf", "anyOf", "oneOf", "items", "prefixItems"}
SUBSCHEMA_MAPS = {"properties", "patternProperties", "definitions", "$defs", "dependencies", "dependentSchemas"}


def catalog_modules(directory=None):
    """Return the module files load_catalog imports from a catalog directory, in load order.

    Raises:
        CatalogLoadError: The directory is missing or cannot be listed (glob would see it empty).
    """
    directory = Path(directory or DEFAULT_CATALOG)
    if not directory.is_dir():
        raise CatalogLoadError(f"ToolSmith catalog {str(directory)!r} is not a directory")
    try:
        names = os.listdir(directory)
    except OSError as exc:
        raise CatalogLoadError(f"ToolSmith catalog {str(directory)!r} cannot be read ({describe(exc)})") from exc
    return [directory / name for name in sorted(names)
            if name.endswith(".py") and not name.startswith(("_", ".")) and (directory / name).is_file()]


def fingerprint_files(files):
    """Return a SHA-256 over LOADER_RULES and (file name, contents) pairs, in the order given."""
    digest = hashlib.sha256(f"toolsmith loader rules {LOADER_RULES}\0".encode())
    for name, data in files:
        digest.update(os.fsencode(name) + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


def catalog_fingerprint(directory=None):
    """Return a SHA-256 over LOADER_RULES and the file names and contents of the modules load_catalog imports.

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
    schema of type "object", since the router passes the arguments to execute(**args) as
    named arguments, without, where a keyword can stand, the keywords that can make validation
    fetch other documents or take exponential time (REFUSED_KEYWORDS); property and
    definition names may look like them. Tuples count as the arrays JSON carries them
    as. Any error while checking is a problem too, so a schema that breaks the checker
    is skipped like an invalid one. tool is a GeneratedTool, or the mapping a worker
    reports for one.
    """
    if not valid_metadata(tool):
        return "tool_name and tool_description must be strings and input_schema a mapping"
    schema = tool["input_schema"] if isinstance(tool, dict) else tool.input_schema
    try:
        schema = _as_json(schema)
        found = sorted({key for key, _ in _keywords(schema) if key in REFUSED_KEYWORDS})
        if found:
            return (f"input_schema uses {', '.join(found)}, which generated tools may not use: such keywords can "
                    "make argument validation fetch documents or take exponential time")
        Draft7Validator.check_schema(schema)
        if schema.get("type") not in ("object", ["object"]):
            found = f"it has {_brief(json.dumps(schema['type']), 60)}" if "type" in schema else "it has none"
            return ('input_schema must have "type": "object", since the router passes the arguments to execute() '
                    f"as named arguments ({found})")
    except SchemaError as exc:
        return f"input_schema is not a valid JSON Schema: {_brief(exc.message)}"
    except RecursionError:
        return "input_schema is nested too deeply"
    except Exception as exc:  # a value the checker itself cannot handle
        return f"input_schema cannot be checked ({describe(exc, 200)})"
    return None


def json_schema(schema):
    """The input schema as the agent and the validator use it: tuples become the lists JSON carries."""
    return _as_json(schema)


def _brief(text, limit=200):
    """text, cut to limit characters."""
    return text if len(text) <= limit else text[:limit] + f"... ({len(text) - limit} more characters)"


def _as_json(value):
    """Return a JSON value with tuples turned into lists, as JSON carries them."""
    if isinstance(value, dict):
        return {key: _as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(item) for item in value]
    return value


def _keywords(schema):
    """Yield (keyword, value) for every keyword of a schema and of the subschemas it holds.

    The names in properties, definitions and the like are not keywords, and neither is
    anything inside enum, const, default, examples or an unknown keyword.
    """
    if not isinstance(schema, dict):
        return
    for key, value in schema.items():
        yield key, value
        if key in SUBSCHEMA_MAPS and isinstance(value, dict):
            for item in value.values():
                yield from _keywords(item)  # a dependencies entry that lists names is no schema
        elif key in SUBSCHEMA_LISTS and isinstance(value, list):
            for item in value:
                yield from _keywords(item)
        elif key in SUBSCHEMA:
            yield from _keywords(value)


def load_catalog(directory=None, *, client=None, strict=False):
    """Import every tool module in a catalog directory and instantiate its tool.

    A tool the agent cannot offer (see tool_problem) is skipped with a warning, so
    one bad module leaves the rest of the catalog usable.

    Args:
        directory: Directory holding one GeneratedTool subclass per .py file;
            files starting with an underscore are skipped. Defaults to the
            packaged catalog.
        client: Optional ToolClient the tools use to call other router tools.
        strict: Raise CatalogLoadError for a tool the agent cannot offer instead of
            skipping it, as Gate 1 does for a candidate.

    Returns:
        A mapping of tool names to tool instances, ordered by file name. Each keeps the
        input_schema it defines, which may be a read-only property; its readers take it
        through json_schema, as the check does.

    Raises:
        CatalogLoadError: The directory is missing, a module fails to import (sys.exit
            included), to build its tool or to take the client, a module does not define
            exactly one tool, two tools share a name, or strict is set and a tool cannot
            be offered. A failing module's own error is the cause, and its message names
            the module.
    """
    tools = {}
    for path in catalog_modules(directory):
        spec = importlib.util.spec_from_file_location(f"toolsmith_catalog_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            classes = [value for value in vars(module).values() if inspect.isclass(value)
                       and issubclass(value, GeneratedTool) and value.__module__ == module.__name__]
            tool = classes[0]() if len(classes) == 1 else None
            if tool is not None:
                tool.client = client
        except (Exception, SystemExit) as exc:
            raise CatalogLoadError(f"the ToolSmith catalog {path.parent} cannot load {path.name}: "
                                   f"{describe(exc)}") from exc
        if tool is None:
            raise CatalogLoadError(f"{path} must define exactly one GeneratedTool subclass")
        problem = tool_problem(tool)
        if problem and strict:
            raise CatalogLoadError(f"{path}: {problem}")
        if problem:
            logger.warning("ToolSmith catalog: skipping %s: %s", path.name, problem)
            continue
        if tool.tool_name in tools:
            raise CatalogLoadError(f"Duplicate generated tool name {tool.tool_name!r} in {path}")
        tools[tool.tool_name] = tool
    return tools
