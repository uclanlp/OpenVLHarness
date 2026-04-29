import contextlib
import os

import server.app as server_app


@contextlib.contextmanager
def _patch_env(updates):
    original = {
        key: os.environ.get(key)
        for key in updates
    }
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_env_snapshot_is_allowlisted_runtime_state_with_redaction():
    with _patch_env(
        {
            "HOST": "127.0.0.1",
            "PORT": "8101",
            "TOOL_MODULES": "tools.python_executor.tool",
            "LLM_API_KEY": "secret-value",
            "UNRELATED_SECRET": "must-not-leak",
        }
    ):
        snapshot = server_app.env_snapshot()

    assert snapshot["HOST"] == "127.0.0.1"
    assert snapshot["PORT"] == "8101"
    assert snapshot["TOOL_MODULES"] == "tools.python_executor.tool"
    assert snapshot["LLM_API_KEY"] == "<REDACTED>"
    assert "UNRELATED_SECRET" not in snapshot
    assert snapshot["_SOURCE"] == "tool_server_runtime"


if __name__ == "__main__":
    test_env_snapshot_is_allowlisted_runtime_state_with_redaction()
    print("Env snapshot tests completed.")
