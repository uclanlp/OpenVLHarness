"""Collect the evaluator tests only when the evaluation extra is installed."""
try:
    import pandas  # noqa: F401
except ImportError:  # pragma: no cover - exercised in minimal installs
    collect_ignore_glob = ["evaluation/*"]
