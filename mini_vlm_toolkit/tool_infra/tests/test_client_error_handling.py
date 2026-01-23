import os

from client.client import ToolClient, ToolClientError
from tests.utils import ensure_tool_server


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")


def log_step(title):
    print(f"\n== {title} ==")


def test_client_error_detail_propagation():
    log_step("Client error detail propagation")
    client = ToolClient(SERVER_URL)
    try:
        client.execute("Python_Executor_Tool", {"code": ""})
    except ToolClientError as exc:
        print(f"Caught error: {exc}")
        assert "code is required" in str(exc), "Expected error detail in message"
        if exc.detail is not None:
            assert "code is required" in str(exc.detail), "Expected error detail field"
        return
    except Exception as exc:
        raise AssertionError(
            f"Expected ToolClientError, got {type(exc).__name__}: {exc}"
        )
    raise AssertionError("Expected client error but request succeeded")


def main():
    if not ensure_tool_server(SERVER_URL, required_tools=["Python_Executor_Tool"]):
        return
    test_client_error_detail_propagation()
    print("Client error handling test completed.")


if __name__ == "__main__":
    main()
