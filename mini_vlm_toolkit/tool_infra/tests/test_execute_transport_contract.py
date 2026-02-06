import contextlib
import datetime
import json

import server.app as server_app
from server.app import ExecuteRequest
from tools.base import BaseTool


class _EchoTool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="Echo_Tool",
            tool_description="Echo input.",
            tool_version="0.1.0",
            input_types={"value": "str"},
            output_type="dict",
        )

    def execute(self, value):
        return {"echo": value}


class _FailingTool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="Failing_Tool",
            tool_description="Always fails.",
            tool_version="0.1.0",
            input_types={"value": "str"},
            output_type="dict",
        )

    def execute(self, value):
        raise ValueError(f"bad input: {value}")


class _NonSerializableTool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="NonSerializable_Tool",
            tool_description="Returns mixed payload types.",
            tool_version="0.1.0",
            input_types={},
            output_type="dict",
        )

    def execute(self):
        return {
            "set_values": {3, 1, 2},
            "raw_bytes": b"\x01\x02",
            "nan_value": float("nan"),
            "dt": datetime.datetime(2024, 1, 1, 9, 30, 0),
        }


@contextlib.contextmanager
def _patch_registry(instances=None, metadata=None):
    original_instances = server_app.tool_instances
    original_metadata = server_app.tool_metadata
    server_app.tool_instances = instances or {}
    server_app.tool_metadata = metadata or {}
    try:
        yield
    finally:
        server_app.tool_instances = original_instances
        server_app.tool_metadata = original_metadata


def test_success_envelope_and_legacy_fields():
    tool = _EchoTool()
    with _patch_registry(
        instances={tool.tool_name: tool},
        metadata={tool.tool_name: tool.get_metadata()},
    ):
        payload, status_code = server_app.execute_tool_request(
            ExecuteRequest(
                tool_name=tool.tool_name,
                args={"value": "hello"},
                tool_call_id="call_123",
            )
        )

    assert status_code == 200
    assert payload["ok"] is True
    assert payload["tool_name"] == tool.tool_name
    assert payload["result"] == {"echo": "hello"}
    assert payload["error"] is None
    assert payload["tool_call_id"] == "call_123"
    assert isinstance(payload["meta"], dict)


def test_error_envelope_tool_not_found():
    with _patch_registry(instances={}, metadata={}):
        payload, status_code = server_app.execute_tool_request(
            ExecuteRequest(tool_name="Missing_Tool", args={})
        )

    assert status_code == 404
    assert payload["ok"] is False
    assert payload["tool_name"] == "Missing_Tool"
    assert payload["result"] is None
    assert payload["error"]["code"] == "tool_not_found"
    assert "Tool not found" in payload["error"]["message"]


def test_error_envelope_execution_failure():
    tool = _FailingTool()
    with _patch_registry(
        instances={tool.tool_name: tool},
        metadata={tool.tool_name: tool.get_metadata()},
    ):
        payload, status_code = server_app.execute_tool_request(
            ExecuteRequest(tool_name=tool.tool_name, args={"value": "x"})
        )

    assert status_code == 500
    assert payload["ok"] is False
    assert payload["tool_name"] == tool.tool_name
    assert payload["result"] is None
    assert payload["error"]["code"] == "tool_execution_error"
    assert "bad input: x" in payload["error"]["message"]
    assert payload["error"]["detail"]["exception_type"] == "ValueError"


def test_success_normalizes_non_json_result():
    tool = _NonSerializableTool()
    with _patch_registry(
        instances={tool.tool_name: tool},
        metadata={tool.tool_name: tool.get_metadata()},
    ):
        payload, status_code = server_app.execute_tool_request(
            ExecuteRequest(tool_name=tool.tool_name, args={})
        )

    assert status_code == 200
    assert payload["ok"] is True
    assert payload["tool_name"] == tool.tool_name
    assert payload["error"] is None
    assert isinstance(payload["meta"], dict)
    assert payload["meta"].get("normalization_warnings")

    # Contract: result must always be JSON-serializable.
    json.dumps(payload["result"], allow_nan=False)


def main():
    test_success_envelope_and_legacy_fields()
    test_error_envelope_tool_not_found()
    test_error_envelope_execution_failure()
    test_success_normalizes_non_json_result()
    print("Execute transport contract tests completed.")


if __name__ == "__main__":
    main()
