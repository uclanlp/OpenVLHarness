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
    explicit_metadata = explicit.get_metadata()
    explicit_metadata["tool_aliases"] = ["explicit_schema_tool"]
    explicit_metadata["is_alias"] = False
    explicit_metadata["canonical_tool_name"] = explicit.tool_name

    explicit_alias_metadata = dict(explicit_metadata)
    explicit_alias_metadata["tool_name"] = "explicit_schema_tool"
    explicit_alias_metadata["is_alias"] = True
    explicit_alias_metadata["canonical_tool_name"] = explicit.tool_name

    with _patch_metadata(
        {
            explicit.tool_name: explicit_metadata,
            "explicit_schema_tool": explicit_alias_metadata,
            fallback.tool_name: fallback.get_metadata(),
        }
    ):
        schemas = server_app._build_tool_jsonschemas()

    explicit_schema = _find_schema(schemas, explicit.tool_name)
    explicit_params = explicit_schema["function"]["parameters"]
    assert explicit_params == explicit.input_schema
    explicit_description = explicit_schema["function"]["description"]

    fallback_schema = _find_schema(schemas, fallback.tool_name)
    fallback_params = fallback_schema["function"]["parameters"]
    assert fallback_params["type"] == "object"
    assert "query" in fallback_params["properties"]
    assert fallback_params["properties"]["query"]["type"] == "string"
    assert "max_results" in fallback_params["properties"]
    assert fallback_params["properties"]["max_results"]["type"] == "integer"
    assert fallback_params["additionalProperties"] is True

    alias_schema = _find_schema(schemas, "explicit_schema_tool")
    alias_params = alias_schema["function"]["parameters"]
    assert alias_params == explicit.input_schema
    assert alias_schema["function"]["description"] == explicit_description


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
            server_app._build_tool_jsonschemas()
        except ValueError as exc:
            assert "input_schema.type must be 'object'" in str(exc)
            return
    raise AssertionError("Expected ValueError for invalid input_schema")


def test_llm_schema_projects_list_and_nested_image_fields():
    metadata = {
        "Synthetic_Spatial_Tool": {
            "tool_name": "Synthetic_Spatial_Tool",
            "tool_description": "Synthetic spatial schema.",
            "input_types": {},
            "input_schema": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "image_path": {"type": "string"},
                            "image_base64": {"type": "string"},
                        },
                    },
                    "image_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "image_base64s": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "_image_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["subject"],
                "additionalProperties": False,
            },
        }
    }
    with _patch_metadata(metadata):
        schemas = server_app._build_tool_llm_schemas()

    schema = _find_schema(schemas, "Synthetic_Spatial_Tool")
    params = schema["function"]["parameters"]
    assert "image_ids" in params["properties"]
    assert "image_base64s" not in params["properties"]
    assert "_image_ids" not in params["properties"]
    subject_props = params["properties"]["subject"]["properties"]
    assert "image_id" in subject_props
    assert "image_path" not in subject_props
    assert "image_base64" not in subject_props


def test_llm_schema_projects_image_paths_plural_to_image_ids():
    """Tracking_Tool-shaped schema: plural `image_paths` must project to `image_ids`."""
    metadata = {
        "Synthetic_Tracking_Tool": {
            "tool_name": "Synthetic_Tracking_Tool",
            "tool_description": "Synthetic plural-image schema.",
            "input_types": {},
            "input_schema": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "image_base64s": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["target"],
                "oneOf": [
                    {"required": ["image_paths"]},
                    {"required": ["image_base64s"]},
                ],
                "additionalProperties": False,
            },
        }
    }
    with _patch_metadata(metadata):
        schemas = server_app._build_tool_llm_schemas()

    schema = _find_schema(schemas, "Synthetic_Tracking_Tool")
    params = schema["function"]["parameters"]
    properties = params["properties"]
    assert "image_ids" in properties
    assert properties["image_ids"]["type"] == "array"
    assert properties["image_ids"]["items"]["type"] == "string"
    assert "image_paths" not in properties
    assert "image_base64s" not in properties
    # oneOf had two branches; the base64 branch is dropped, leaving a single
    # branch that gets folded into required.
    assert "oneOf" not in params
    assert "image_ids" in params["required"]
    assert "target" in params["required"]


def main():
    test_jsonschema_uses_explicit_schema_when_available()
    test_invalid_input_schema_raises()
    test_llm_schema_projects_list_and_nested_image_fields()
    test_llm_schema_projects_image_paths_plural_to_image_ids()
    print("JSON schema generation tests completed.")


if __name__ == "__main__":
    main()
