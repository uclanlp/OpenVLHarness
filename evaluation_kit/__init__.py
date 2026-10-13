"""OpenVLHarness evaluation layer: benchmark loaders, judges and the inference runner."""

__all__ = ["infer_image_job"]


def __getattr__(name):
    # Imported on first use so `openvlharness-eval` can report a missing eval extra cleanly.
    if name == "infer_image_job":
        from .runner import infer_image_job
        return infer_image_job
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
