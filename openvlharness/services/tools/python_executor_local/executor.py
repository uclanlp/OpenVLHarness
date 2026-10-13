"""Execute generated Python locally using validated, copied context bindings.

Execution runs in a killable subprocess with bounded output and variable
previews, in a temporary working directory and without credential-like
environment variables. Normal Python builtins are available; this is not a
security sandbox.
"""

import copy
import keyword
import math
import re
import signal
import sys
import threading
from dataclasses import dataclass
import json
import os
import pickle
import reprlib
import selectors
import subprocess
import tempfile
from itertools import islice
from pathlib import Path

import numpy as np


RESERVED_CONTEXT_NAMES = {"__builtins__"}
MAX_OUTPUT_BYTES = 65536
MAX_VARIABLES = 64
MAX_VARIABLE_PREVIEW = 2048
MAX_VARIABLE_TOTAL = 16384
TRUNCATION_MARKER = "... (truncated)"
_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|^AWS_|^GOOGLE_APPLICATION", re.IGNORECASE)


class CodeExecutionTimeoutError(Exception):
    """Raised when the active code-execution deadline is reached."""
    pass


@dataclass
class PreparedExecutionContext:
    """Validated binding values to copy into a generated-code execution namespace."""
    bindings: dict


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


class _PreviewRenderer(reprlib.Repr):
    """Bound container previews while retaining dictionary insertion order."""

    def repr_dict(self, value, level):
        """Render a bounded prefix of key/value pairs without sorting bindings."""
        if not value:
            return "{}"
        if level <= 0:
            return "{...}"
        items = [f"{self.repr1(key, level - 1)}: {self.repr1(item, level - 1)}"
                 for key, item in islice(value.items(), self.maxdict)]
        if len(value) > self.maxdict:
            items.append("...")
        return "{" + ", ".join(items) + "}"


def _safe_repr(value, max_length=MAX_VARIABLE_PREVIEW):
    """Render bounded previews of strings, containers, arrays and custom objects."""
    renderer = _PreviewRenderer()
    renderer.maxstring = renderer.maxother = max_length
    renderer.maxlist = renderer.maxtuple = renderer.maxdict = 20
    try:
        with np.printoptions(threshold=50, edgeitems=3):
            representation = renderer.repr(value)
    except Exception:
        representation = f"<repr_error {type(value).__name__}>"
    if len(representation) > max_length:
        representation = representation[:max_length - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER
    return representation


def _is_reserved_context_name(name):
    """Recognize builtins and double-underscore names disallowed as injected bindings."""
    if name in RESERVED_CONTEXT_NAMES:
        return True
    return name.startswith("__") and name.endswith("__") and len(name) > 4


def _validate_json_compatible(value, path):
    """Validate nested JSON-like values or NumPy arrays, raising on unsupported values."""
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
    """Validate execution variable names and values without copying nested data.

    Returns an independent top-level mapping; None becomes an empty mapping.
    Raises TypeError for unsupported structures and ValueError for invalid names
    or non-finite scalar floats. NumPy arrays are accepted directly.
    """
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
    """Wrap validated bindings for execution; execute_python_code copies their values."""
    return PreparedExecutionContext(
        bindings=validate_context_bindings(context_bindings),
    )


def _render_namespace_variables(namespace):
    """Bound the number, individual length and total size of variable previews."""
    rendered, remaining = {}, MAX_VARIABLE_TOTAL
    for key, value in namespace.items():
        if key.startswith("__"):
            continue
        if len(rendered) >= MAX_VARIABLES or remaining < MAX_VARIABLE_PREVIEW:
            rendered["__truncated__"] = TRUNCATION_MARKER
            break
        name = key[:128]
        reserve = len("__truncated__") + len(TRUNCATION_MARKER)
        preview = _safe_repr(value, min(MAX_VARIABLE_PREVIEW, remaining - len(name) - reserve))
        rendered[name] = preview
        remaining -= len(name) + len(preview)
    return rendered


def sanitize_code(code):
    """Remove exit/quit calls from generated code."""
    for func in ("exit", "quit", "sys.exit"):
        pattern = rf"{re.escape(func)}\s*\([^)]*\)"
        code = re.sub(pattern, "None", code)
    return code


def _worker_main(result_path):
    """Execute one serialized request and write only JSON diagnostics to its file."""
    try:
        code, bindings = pickle.load(sys.stdin.buffer)
        namespace = copy.deepcopy(bindings)
        exec(code, namespace, namespace)
        result = {"variables": _render_namespace_variables(namespace)}
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            message = "SystemExit is not allowed in executed code"
        else:
            try:
                message = str(exc)[:MAX_VARIABLE_PREVIEW]
            except BaseException:
                message = "Unable to render execution exception"
        result = {"error": message, "error_type": type(exc).__name__}
    Path(result_path).write_text(json.dumps(result), encoding="utf-8")


def _kill_process_group(process):
    """Stop the worker and any children that retain its POSIX process group."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except ProcessLookupError:
        pass


def execute_python_code(code, prepared_context, timeout_seconds=10):
    """Execute Python in a disposable subprocess with copied NumPy bindings.

    Args:
        code: Nonempty Python text, optionally in a Markdown code fence.
        prepared_context: Validated bindings to serialize into the child process.
        timeout_seconds: Positive finite deadline covering startup, execution and
            variable rendering. On expiry the process group is forcibly killed.

    Returns:
        Stripped printed_output (at most 64 KiB including a truncation marker),
        bounded repr-based variables and the extracted/sanitized execution_code.
        Stdout, including native writes, is captured only from the child process.

    Raises:
        ValueError: Code is empty or the timeout is not positive and finite.
        CodeExecutionTimeoutError: The child exceeds the deadline.
        RuntimeError: Execution fails or the child exits without a result.

    Bindings are serialized locally, retaining NumPy dtype and shape. The child
    runs in a temporary working directory, and environment variables whose names
    look like credentials are removed. Generated code otherwise has normal process
    permissions, including network and filesystem access; process isolation is
    not a sandbox.
    """
    if not code:
        raise ValueError("code is required")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    exec_code = sanitize_code(extract_python_code(code))
    captured = bytearray()
    truncated = False
    stop_reader = threading.Event()

    def drain(stream):
        nonlocal truncated
        with stream, selectors.DefaultSelector() as selector:
            selector.register(stream, selectors.EVENT_READ)
            while not stop_reader.is_set():
                if not selector.select(timeout=0.1):
                    continue
                chunk = os.read(stream.fileno(), 8192)
                if not chunk:
                    break
                room = MAX_OUTPUT_BYTES - len(captured)
                captured.extend(chunk[:room])
                truncated = truncated or len(chunk) > room

    with tempfile.TemporaryDirectory(prefix="openvlharness-python-") as temp:
        result_path = Path(temp) / "result.json"
        with tempfile.TemporaryFile() as request:
            pickle.dump((exec_code, prepared_context.bindings), request, protocol=pickle.HIGHEST_PROTOCOL)
            request.seek(0)
            environment = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
            environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(result_path)],
                stdin=request, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                start_new_session=True, env=environment, cwd=temp)
            reader = threading.Thread(target=drain, args=(process.stdout,), daemon=True)
            reader.start()
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise CodeExecutionTimeoutError("Execution timed out") from exc
            finally:
                _kill_process_group(process)
                process.wait()
                reader.join(timeout=1)
                stop_reader.set()
                reader.join()
        if not result_path.is_file():
            raise RuntimeError(f"Python worker exited without a result (exit code {process.returncode})")
        result = json.loads(result_path.read_text(encoding="utf-8"))
    if "error" in result:
        raise RuntimeError(result["error"])
    output = bytes(captured)
    if truncated:
        marker = ("\n" + TRUNCATION_MARKER).encode()
        output = output[:MAX_OUTPUT_BYTES - len(marker)] + marker
    return {"printed_output": output.decode("utf-8", errors="replace").strip(),
            "variables": result["variables"], "execution_code": exec_code}


if __name__ == "__main__":
    _worker_main(sys.argv[1])
