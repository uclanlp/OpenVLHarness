from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import types

repo_root = Path(__file__).resolve().parents[3]
pkg_root = repo_root / "mini_vlm_toolkit"

# Avoid importing mini_vlm_toolkit/agent/__init__.py, which pulls optional deps.
mini_pkg = types.ModuleType("mini_vlm_toolkit")
mini_pkg.__path__ = [str(pkg_root)]
sys.modules.setdefault("mini_vlm_toolkit", mini_pkg)

agent_pkg = types.ModuleType("mini_vlm_toolkit.agent")
agent_pkg.__path__ = [str(pkg_root / "agent")]
sys.modules.setdefault("mini_vlm_toolkit.agent", agent_pkg)

from mini_vlm_toolkit.agent.code_execution import (  # noqa: E402
    CodeExecutionTimeoutError,
    execute_python_code,
    prepare_execution_context,
)


def test_worker_thread_python_loop_times_out():
    def run_timeout_case():
        prepared = prepare_execution_context({})
        try:
            execute_python_code(
                "while True:\n    pass",
                prepared,
                timeout_seconds=0.1,
            )
        except CodeExecutionTimeoutError:
            return "timed_out"
        return "completed"

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(run_timeout_case).result(timeout=3) == "timed_out"


def test_worker_thread_timeout_restores_trace_hook():
    def run_timeout_case():
        prepared = prepare_execution_context({})
        try:
            execute_python_code(
                "while True:\n    pass",
                prepared,
                timeout_seconds=0.1,
            )
        except CodeExecutionTimeoutError:
            return "timed_out"
        return "completed"

    def run_success_case():
        prepared = prepare_execution_context({})
        return execute_python_code(
            "value = 2 + 2\nprint(value)",
            prepared,
            timeout_seconds=1,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(run_timeout_case).result(timeout=3) == "timed_out"
        result = executor.submit(run_success_case).result(timeout=3)

    assert result["printed_output"] == "4"
    assert result["variables"]["value"] == "4"


if __name__ == "__main__":
    test_worker_thread_python_loop_times_out()
    print("PASS: test_worker_thread_python_loop_times_out")
    test_worker_thread_timeout_restores_trace_hook()
    print("PASS: test_worker_thread_timeout_restores_trace_hook")
