"""Process-wide counters for silent degradations.

A degradation is a path where a tool still returns a usable result, but a
component it depends on failed and the tool fell back to something weaker,
for example the summarizer timed out and Text_Search returned raw results
instead of a summary. Such fallbacks are invisible in HTTP status codes and
in in-flight counts, so every one of them is counted here and exposed by the
router's ``/backend-status`` endpoint.

Counters are process-local and monotonic; a restart resets them.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Dict

_LOCK = threading.Lock()
_COUNTS: Dict[str, int] = defaultdict(int)
_LAST: Dict[str, Dict[str, Any]] = {}


def record(kind: str, *, tool: str = "", reason: str = "") -> None:
    """Count one degradation.

    Args:
        kind: Coarse bucket such as ``"summarize"``; keep the cardinality low.
        tool: The tool that degraded, for attribution.
        reason: Short machine-readable reason such as ``"APITimeoutError"``.
    """
    with _LOCK:
        _COUNTS[kind] += 1
        if reason:
            _COUNTS[f"{kind}:{reason}"] += 1
        _LAST[kind] = {"tool": tool, "reason": reason, "at": time.time()}


def snapshot() -> Dict[str, Any]:
    """Return the current counters in a JSON-serialisable form."""
    with _LOCK:
        return {"counts": dict(_COUNTS), "last": dict(_LAST)}


def reset() -> None:
    """Clear all counters (tests only)."""
    with _LOCK:
        _COUNTS.clear()
        _LAST.clear()
