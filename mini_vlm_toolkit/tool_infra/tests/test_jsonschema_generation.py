import contextlib

import server.app as server_app
from tools.base import BaseTool


class _ExplicitSchemaTool(BaseTool):
    def __init__(self):
        input_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["query"],
            "additionalProperties": False,
        }
        super().__init__(
            tool_name="ExplicitSchema_Tool",
            tool_description="Tool with explicit input schema.",
            tool_version="0.1.0",
            input_types={"query": "str", "limit": "int"},
            input_schema=input_schema,
            output_type="dict",
        )


class _FallbackSchemaTool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="FallbackSchema_Tool",
            tool_description="Tool without explicit schema.",
            tool_version="0.1.0",
            input_types={
                "query": "str - search query",
                "max_results": "int - max results",
            },
            output_type="dict",
        )


@contextlib.contextmanager
def _patch_metadata(metadata):
    original_metadata = server_app.tool_metadata
    server_app.tool_metadata = metadata
    try:
        yield
    finally:
        server_app.tool_metadata = original_metadata


def _find_schema(schemas, tool_name):
    for schema in schemas:
        function_schema = schema.get("function", {})
        if function_schema.get("name") == tool_name:
            return schema
    raise AssertionError(f"Schema not found for {tool_name}")


def test_jsonschema_uses_explicit_schema_when_available():
    explicit = _ExplicitSchemaTool()
    fallback = _FallbackSchemaTool()
    with _patch_metadata(
        {
            explicit.tool_name: explicit.get_metadata(),
            fallback.tool_name: fallback.get_metadata(),
        }
    ):
        schemas = server_app.list_tools_jsonschema()

    explicit_schema = _find_schema(schemas, explicit.tool_name)
    explicit_params = explicit_schema["function"]["parameters"]
    assert explicit_params == explicit.input_schema

    fallback_schema = _find_schema(schemas, fallback.tool_name)
    fallback_params = fallback_schema["function"]["parameters"]
    assert fallback_params["type"] == "object"
    assert "query" in fallback_params["properties"]
    assert fallback_params["properties"]["query"]["type"] == "string"
    assert "max_results" in fallback_params["properties"]
    assert fallback_params["properties"]["max_results"]["type"] == "integer"
    assert fallback_params["additionalProperties"] is True


def test_invalid_input_schema_raises():
    invalid_metadata = {
        "InvalidSchema_Tool": {
            "tool_name": "InvalidSchema_Tool",
            "tool_description": "Broken schema",
            "input_types": {"value": "str"},
            "input_schema": {
                "type": "array",
                "items": {"type": "string"},
            },
        }
    }
    with _patch_metadata(invalid_metadata):
        try:
            server_app.list_tools_jsonschema()
        except ValueError as exc:
            assert "input_schema.type must be 'object'" in str(exc)
            return
    raise AssertionError("Expected ValueError for invalid input_schema")


def main():
    test_jsonschema_uses_explicit_schema_when_available()
    test_invalid_input_schema_raises()
    print("JSON schema generation tests completed.")


if __name__ == "__main__":
    main()
