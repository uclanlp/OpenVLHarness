"""Errors that keep ToolSmith from starting, which the command lines report in one line.

The router, the evaluator and the pipeline print a ToolSmithStartError as "error: ..."
without a traceback, because its message says what to change; any other error keeps
its traceback.
"""


class ToolSmithStartError(Exception):
    """ToolSmith cannot start: its catalog, its worker or its sentence encoder cannot load."""


class CatalogLoadError(ToolSmithStartError, ValueError):
    """A catalog that does not load: a module that fails to import, or tools that clash."""


class WorkerStartError(ToolSmithStartError, RuntimeError):
    """The generated-tool worker cannot run or load its catalog: no user namespaces, a timeout, a crash."""


class EncoderError(ToolSmithStartError, RuntimeError):
    """The sentence encoder cannot be loaded; the message says what to change."""


def describe(exc, limit=2000):
    """Name an exception and its message, cut to limit characters; for any exception, also one that cannot be printed."""
    try:
        kind = type(exc).__name__
        kind = kind if isinstance(kind, str) else "Exception"
    except Exception:
        kind = "Exception"
    try:
        text = str(exc)
    except Exception:
        text = "<unprintable message>"
    if len(text) > limit:
        text = text[:limit] + f"... ({len(text) - limit} more characters)"
    return f"{kind}: {text}" if text else kind
