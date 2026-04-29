import base64
import copy
import contextlib
import keyword
import math
import re
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO

import numpy as np


RESERVED_CONTEXT_NAMES = {"__builtins__"}
BASE64_LIKE_PATTERN = re.compile(r"^[A-Za-z0-9+/=\s]+$")
DATA_URL_PATTERN = re.compile(r"^data:([^;,]+)?(?:;base64)?,", re.IGNORECASE)
TRANSPORT_TYPE_KEY = "__toolvision_type__"
TRANSPORT_NDARRAY_TYPE = "ndarray"


class TimeoutError(Exception):
    pass


@dataclass
class PreparedExecutionContext:
    bindings: dict


@contextmanager
def timeout(seconds):
    if not seconds or seconds <= 0:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise TimeoutError("Execution timed out")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def extract_python_code(text):
    match = re.search(
        r"```\s*(?:python|py)\b\s*(.*?)\s*```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def truncate_string(text, max_length):
    if text is None:
        return ""
    if max_length is None or max_length <= 0 or len(text) <= max_length:
        return text

    marker = " ... (truncated: middle content omitted) ... "
    if max_length <= len(marker) + 2:
        return text[:max_length]

    head_size = (max_length - len(marker)) // 2
    tail_size = max_length - len(marker) - head_size
    return f"{text[:head_size]}{marker}{text[-tail_size:]}"


def safe_repr(value, max_length=None):
    try:
        representation = repr(value)
    except Exception as exc:
        representation = f"<repr_error {type(value).__name__}: {exc}>"
    else:
        if not isinstance(representation, str):
            representation = f"<repr_error {type(value).__name__}: non-string repr>"
    return truncate_string(representation, max_length)


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


def _decode_transport_value(value):
    if isinstance(value, dict):
        if value.get(TRANSPORT_TYPE_KEY) == TRANSPORT_NDARRAY_TYPE:
            raw = base64.b64decode(value["data_b64"])
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(dim) for dim in value["shape"])
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        return {
            key: _decode_transport_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_decode_transport_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_transport_value(item) for item in value)
    return value


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
        # Tagged ndarray payloads are a private coding-tool transport detail
        # for agent-injected context_bindings over /execute. They are decoded
        # here so Python_Executor_Tool and Python_Code_Generator_Tool receive
        # normal ndarray values; this is not a general format for other tools.
        validated[key] = _decode_transport_value(value)
    return validated


def prepare_execution_context(context_bindings=None):
    return PreparedExecutionContext(
        bindings=validate_context_bindings(context_bindings),
    )


def _looks_like_data_url(value):
    return isinstance(value, str) and DATA_URL_PATTERN.match(value) is not None


def _get_data_url_media_type(value):
    match = DATA_URL_PATTERN.match(value)
    if match is None:
        return "unknown"
    return match.group(1) or "text/plain"


def _looks_like_base64(value):
    if not isinstance(value, str):
        return False
    stripped = "".join(value.split())
    if len(stripped) < 128:
        return False
    return BASE64_LIKE_PATTERN.fullmatch(stripped) is not None


def _simple_type_name(value):
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _infer_item_hint(values):
    if not values:
        return ""
    sample = list(values[:3])
    type_names = []
    for item in sample:
        name = _simple_type_name(item)
        if name not in type_names:
            type_names.append(name)
    if len(type_names) == 1:
        return type_names[0]
    return "mixed[" + ", ".join(type_names) + "]"


def summarize_binding_type(value):
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        if _looks_like_data_url(value):
            media_type = _get_data_url_media_type(value)
            return f"str data-url {media_type} len={len(value)}"
        if _looks_like_base64(value):
            return f"str base64-like len={len(value)}"
        return f"str len={len(value)}"
    if isinstance(value, list):
        item_hint = _infer_item_hint(value)
        suffix = f" item_hint={item_hint}" if item_hint else ""
        return f"list len={len(value)}{suffix}"
    if isinstance(value, tuple):
        item_hint = _infer_item_hint(list(value))
        suffix = f" item_hint={item_hint}" if item_hint else ""
        return f"tuple len={len(value)}{suffix}"
    if isinstance(value, dict):
        return f"dict len={len(value)}"
    return type(value).__name__


def build_context_prompt_sections(
    prepared_context,
    binding_preview_length=160,
):
    sections = []

    if prepared_context.bindings:
        lines = [
            "Available runtime bindings already injected by the harness:"
        ]
        for name, value in prepared_context.bindings.items():
            type_summary = summarize_binding_type(value)
            preview = safe_repr(value, max_length=binding_preview_length)
            line = f"- {name}: {type_summary}"
            if preview:
                line += f" value_preview={preview}"
            lines.append(line)
        sections.append("\n".join(lines))

    return "\n\n".join(section for section in sections if section.strip())


def build_rich_data_catalog(binding_descriptors):
    """Render binding descriptors into a rich Python-comment-style catalog.

    Args:
        binding_descriptors: dict of serialized BindingDescriptor dicts,
            as produced by ToolContextMemory.get_binding_descriptors_payload().

    Returns:
        A multi-line string suitable for inclusion in a coding LLM prompt.
    """
    if not binding_descriptors:
        return ""

    lines = ["# Available variables (already in scope, use directly):"]

    for _key, desc in binding_descriptors.items():
        key = desc.get("key", _key)
        source_tool = desc.get("source_tool", "")
        source_args = desc.get("source_args", {})
        fields = desc.get("fields", {})

        # Header line with provenance
        source_parts = []
        if source_tool:
            arg_str = ", ".join(
                f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                for k, v in source_args.items()
            )
            source_parts.append(
                f"{source_tool}({arg_str})" if arg_str else source_tool
            )
        source_suffix = f" -- from {source_parts[0]}" if source_parts else ""
        lines.append(f"")
        lines.append(f"# {key}{source_suffix}")

        # Field lines
        for fname, finfo in fields.items():
            if isinstance(finfo, dict):
                dtype = finfo.get("dtype", "")
                description = finfo.get("description", "")
                preview = finfo.get("preview", "")
            else:
                dtype = description = preview = ""

            parts = [f'#   ["{fname}"]: {dtype}']
            if description:
                parts[0] += f" -- {description}"
            lines.append(parts[0])
            if preview:
                lines.append(f"#     preview: {preview}")

    return "\n".join(lines)


def render_namespace_variables(namespace):
    visible_items = [
        (key, value)
        for key, value in namespace.items()
        if not key.startswith("__")
    ]
    rendered = {}
    for key, value in visible_items:
        rendered[key] = safe_repr(value)
    return rendered


def execute_python_code(
    code,
    prepared_context,
    timeout_seconds=10,
    sanitize_main_code=None,
):
    if not code:
        raise ValueError("code is required")

    exec_code = extract_python_code(code)
    if sanitize_main_code is not None:
        exec_code = sanitize_main_code(exec_code)

    namespace = copy.deepcopy(prepared_context.bindings)
    stdout_buffer = StringIO()

    try:
        with timeout(timeout_seconds):
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
        "variables": render_namespace_variables(namespace),
        "execution_code": exec_code,
    }
