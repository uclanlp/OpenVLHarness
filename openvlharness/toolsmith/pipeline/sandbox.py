"""Gate 1 sandbox: load one candidate, run it once on dummy input and print its metadata.

Usage: python -m openvlharness.toolsmith.pipeline.sandbox DIRECTORY
The process cannot write files, is capped at 2 GiB of address space, and every
tool the candidate calls answers with a failure, so the run reaches no service.
"""
import json
import resource
import sys

from ..catalog import load_catalog

IMAGE = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
VALUES = {"string": "test", "integer": 1, "number": 1.0, "boolean": False, "array": [], "object": {}}


def dummy_arguments(schema):
    """Return minimal values for the required parameters of an input schema; images get a 1x1 PNG."""
    arguments = {}
    for key in schema.get("required", []):
        if key.endswith("image_id"):
            arguments[f"{key[:-3]}_base64"] = IMAGE
        else:
            kind = (schema.get("properties", {}).get(key) or {}).get("type", "string")
            arguments[key] = VALUES.get(kind, "test")
    return arguments


class Unavailable:
    """Tool client whose every call fails."""

    def execute(self, tool_name, args, *, tool_call_id):
        """Return a failure envelope for any tool."""
        return {"ok": False, "tool_name": tool_name, "tool_call_id": tool_call_id, "result": None, "meta": {},
                "error": {"code": "unavailable", "message": "tools are unavailable during the sanity run"}}


def main(directory):
    """Run the directory's single tool once and print its name, description and input schema."""
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    tool = next(iter(load_catalog(directory, client=Unavailable()).values()))
    json.dumps(tool.execute(**dummy_arguments(tool.input_schema)))
    print(json.dumps({"tool_name": tool.tool_name, "tool_description": tool.tool_description,
                      "input_schema": tool.input_schema}))


if __name__ == "__main__":
    main(*sys.argv[1:])
