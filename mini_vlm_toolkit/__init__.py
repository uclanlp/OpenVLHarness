try:
    from .inference_video import infer_video_job
except Exception:  # pragma: no cover - optional runtime dependency (e.g., torch)
    infer_video_job = None

try:
    from .inference_image import infer_image_job
except Exception:  # pragma: no cover - optional runtime dependency (e.g., torch)
    infer_image_job = None
