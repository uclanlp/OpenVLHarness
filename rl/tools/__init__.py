"""Tool adapters for verl multi-turn rollout."""

__all__ = ["MiniVLMHTTPTool"]


def __getattr__(name):
    if name == "MiniVLMHTTPTool":
        from .tool_server_adapter import MiniVLMHTTPTool

        return MiniVLMHTTPTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
