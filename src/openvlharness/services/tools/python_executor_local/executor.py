"""Orchestrator-side Python code execution utilities.

Standalone module for executing LLM-generated code with injected bindings.
Adapted from tool_infra/src/tools/python_execution_common.py so the agent
can execute code locally without sending large payloads to the tool server.
"""

import contextlib
import copy
import keyword
import math
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass
from io import StringIO

import numpy as np


RESERVED_CONTEXT_NAMES = {"__builtins__"}


class CodeExecutionTimeoutError(Exception):
    pass


@dataclass
class PreparedExecutionContext:
    bindings: dict


@contextlib.contextmanager
def _trace_timeout(seconds):
    """Current-thread timeout fallback for worker-thread code execution.

    Benchmark inference runs model.generate in ThreadPoolExecutor workers.
    SIGALRM cannot be installed there, so use a per-thread trace hook to keep
    generated Python loops from hanging collection forever. This only traces
    the current worker thread and is restored immediately after execution.
    """
    deadline = time.monotonic() + seconds
    previous_trace = sys.gettrace()

    def trace(frame, event, arg):
        if time.monotonic() >= deadline:
            raise CodeExecutionTimeoutError("Execution timed out")
        return trace

    sys.settrace(trace)
    try:
        yield
    finally:
        sys.settrace(previous_trace)


@contextlib.contextmanager
def _timeout(seconds):
    """Timeout context manager for generated code execution."""
    if not seconds or seconds <= 0:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        with _trace_timeout(seconds):
            yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise CodeExecutionTimeoutError("Execution timed out")

    previous_handler = signal.signal(signal.SIGALRM, handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.monotonic()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL,
                max(previous_timer[0] - (time.monotonic() - started), 0.000001), previous_timer[1])


def extract_python_code(text):
    """Extract Python code from markdown fenced blocks, or return as-is."""
    match = re.search(
        r"```\s*(?:python|py)\b\s*(.*?)\s*```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def _safe_repr(value, max_length=None):
    try:
        representation = repr(value)
    except Exception as exc:
        representation = f"<repr_error {type(value).__name__}: {exc}>"
    else:
        if not isinstance(representation, str):
            representation = f"<repr_error {type(value).__name__}: non-string repr>"
    if max_length and len(representation) > max_length:
        marker = " ... (truncated) ... "
        head = (max_length - len(marker)) // 2
        tail = max_length - len(marker) - head
        return f"{representation[:head]}{marker}{representation[-tail:]}"
    return representation


def _is_reserved_context_name(name):
    if name in RESERVED_CONTEXT_NAMES:
        return True
    return name.startswith("__") and name.endswith("__") and len(name) > 4


def _validate_json_compatible(value, path):
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, np.ndarray):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite floats")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_compatible(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a dict with a non-string key")
            _validate_json_compatible(item, f"{path}.{key}")
        return
    raise TypeError(
        f"{path} contains a non-JSON-serializable value of type {type(value).__name__}"
    )


def validate_context_bindings(context_bindings):
    if context_bindings is None:
        return {}
    if not isinstance(context_bindings, dict):
        raise TypeError("context_bindings must be a dict")

    validated = {}
    for key, value in context_bindings.items():
        if not isinstance(key, str):
            raise TypeError("context_bindings keys must be strings")
        if not key.isidentifier():
            raise ValueError(
                f"context_bindings key {key!r} must be a valid Python identifier"
            )
        if keyword.iskeyword(key):
            raise ValueError(
                f"context_bindings key {key!r} must not be a Python keyword"
            )
        if _is_reserved_context_name(key):
            raise ValueError(
                f"context_bindings key {key!r} is reserved and cannot be used"
            )
        _validate_json_compatible(value, f"context_bindings.{key}")
        validated[key] = value
    return validated


def prepare_execution_context(context_bindings=None):
    return PreparedExecutionContext(
        bindings=validate_context_bindings(context_bindings),
    )


def _render_namespace_variables(namespace):
    visible_items = [
        (key, value)
        for key, value in namespace.items()
        if not key.startswith("__")
    ]
    rendered = {}
    for key, value in visible_items:
        rendered[key] = _safe_repr(value)
    return rendered


def sanitize_code(code):
    """Remove exit/quit calls from generated code."""
    for func in ("exit", "quit", "sys.exit"):
        pattern = rf"{re.escape(func)}\s*\([^)]*\)"
        code = re.sub(pattern, "None", code)
    return code


# redirect_stdout changes process-global state. Concurrent benchmark questions
# must not interleave their capture contexts or leak stdout into another result.
_EXECUTION_LOCK = threading.RLock()


def execute_python_code(
    code,
    prepared_context,
    timeout_seconds=10,
):
    """Execute Python code with injected context bindings.

    Args:
        code: Python code string (may be wrapped in markdown fences).
        prepared_context: PreparedExecutionContext with validated bindings.
        timeout_seconds: Maximum execution time.

    Returns:
        dict with printed_output, variables, and execution_code.
    """
    if not code:
        raise ValueError("code is required")

    exec_code = extract_python_code(code)
    exec_code = sanitize_code(exec_code)

    namespace = copy.deepcopy(prepared_context.bindings)
    stdout_buffer = StringIO()

    try:
        with _EXECUTION_LOCK, _timeout(timeout_seconds):
            with contextlib.redirect_stdout(stdout_buffer):
                exec(exec_code, namespace, namespace)
    except SystemExit as exc:
        if exc.code is None:
            message = "SystemExit is not allowed in executed code"
        else:
            message = f"SystemExit is not allowed in executed code: {exc.code!r}"
        raise RuntimeError(message) from exc

    return {
        "printed_output": stdout_buffer.getvalue().strip(),
        "variables": _render_namespace_variables(namespace),
        "execution_code": exec_code,
    }
