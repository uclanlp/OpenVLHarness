"""Gate 1 sandbox: load one candidate, run it once on dummy input and print its metadata.

Usage: python -m openvlharness.toolsmith.pipeline.sandbox DIRECTORY
The process cannot write files, is capped at 2 GiB of address space, and every
tool the candidate calls answers with a failure, so the run reaches no service.
"""
import json
import os
import sys

from ..catalog import json_schema, load_catalog
from ..errors import CatalogLoadError
from ..worker import limit_resources

IMAGE = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
VALUES = {"string": "test", "integer": 1, "number": 1.0, "boolean": False, "array": [], "object": {}}


def dummy_arguments(schema):
    """Return minimal values for the required parameters of an input schema; images get a 1x1 PNG.

    A type union such as ["string", "null"] uses its first supported member, and null
    only when it is the sole choice.
    """
    arguments = {}
    for key in schema.get("required", []):
        if key.endswith("image_id"):
            arguments[f"{key[:-3]}_base64"] = IMAGE
            continue
        kind = (schema.get("properties", {}).get(key) or {}).get("type", "string")
        if isinstance(kind, list):
            kind = next((member for member in kind if isinstance(member, str) and member in VALUES),
                        "null" if "null" in kind else "string")
        arguments[key] = None if kind == "null" else VALUES.get(kind, "test") if isinstance(kind, str) else "test"
    return arguments


class Unavailable:
    """Tool client whose every call fails."""

    def execute(self, tool_name, args, *, tool_call_id):
        """Return a failure envelope for any tool."""
        return {"ok": False, "tool_name": tool_name, "tool_call_id": tool_call_id, "result": None, "meta": {},
                "error": {"code": "unavailable", "message": "tools are unavailable during the sanity run"}}


def _exits(exc):
    """Fail with what a sys.exit() at import means; it ends the process with no traceback, or as a success."""
    detail = "SystemExit" if exc.code is None else f"SystemExit: {exc.code}"
    print(f"candidate.py exits at import ({detail})" if exc.code in (0, None) else detail, file=sys.stderr)
    sys.exit(1)


def main(directory):
    """Run the directory's single tool once and print its name, description and input schema."""
    limit_resources(memory_bytes=2 << 30, file_bytes=0)
    try:
        tools = load_catalog(directory, client=Unavailable(), strict=True)
    except CatalogLoadError as exc:
        # Gate 1's feedback is the candidate's own error ("NameError: ..."), the same on every run: not
        # the loader's wrapper, which names the temporary directory the candidate runs from.
        if isinstance(exc.__cause__, SystemExit):
            _exits(exc.__cause__)
        if exc.__cause__ is not None:
            raise exc.__cause__ from None
        print(f"ValueError: {str(exc).replace(directory.rstrip(os.sep) + os.sep, '')}", file=sys.stderr)
        sys.exit(1)
    tool = next(iter(tools.values()))
    schema = json_schema(tool.input_schema)  # tuples as the lists the agent validates with
    json.dumps(tool.execute(**dummy_arguments(schema)))
    print(json.dumps({"tool_name": tool.tool_name, "tool_description": tool.tool_description,
                      "input_schema": schema}))


if __name__ == "__main__":
    main(*sys.argv[1:])
