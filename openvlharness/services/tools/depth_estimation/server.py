"""Serve DA3 metric depth and ordered camera poses over FastAPI.

Importing this worker loads the configured model and allocates its device.
The default DA3NESTED-GIANT-LARGE model returns depth in meters.
"""
import base64
import io
import os
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

"""
Monocular Metric Depth: To obtain metric depth in meters from DA3METRIC-LARGE,
use metric_depth = focal * net_output / 300., where focal is the focal length in pixels
(typically the average of fx and fy from the camera intrinsic matrix K).
Note that the output from DA3NESTED-GIANT-LARGE is already in meters.
"""

MODEL_ID = os.getenv(
    "DEPTH_ESTIMATION_MODEL_ID", "depth-anything/DA3NESTED-GIANT-LARGE"
)
DEVICE_ENV = os.getenv("DEPTH_ESTIMATION_DEVICE", "auto")
COLORMAP_DEFAULT = os.getenv("DEPTH_ESTIMATION_COLORMAP", "inferno")

app = FastAPI(title="Depth Estimation Server (DA3)", version="0.1")


def _resolve_device(device):
    """Resolve an explicit device or choose available CUDA, falling back to CPU."""
    if device and device != "auto":
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    """Load the configured DA3 model onto its resolved device and return both."""
    from depth_anything_3.api import DepthAnything3

    device = _resolve_device(DEVICE_ENV)
    model = DepthAnything3.from_pretrained(MODEL_ID)
    model = model.to(device)
    return model, device


MODEL, DEVICE = _load_model()


def _normalize_depth(depth):
    """Scale a depth array to [0, 1], returning zeros for a constant array."""
    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    if depth_max - depth_min < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return (depth - depth_min) / (depth_max - depth_min)


def _apply_colormap(depth_norm, colormap_name):
    """Return uint8 RGB depth colors and their name; use grayscale if matplotlib is unavailable."""
    colormap_name = (colormap_name or "inferno").lower()
    if colormap_name == "inferno":
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.cm as cm
        except Exception:
            colormap_name = "grayscale"
        else:
            colors = cm.get_cmap("inferno")(depth_norm)[:, :, :3]
            return (colors * 255).astype(np.uint8), "inferno"

    if colormap_name == "grayscale":
        gray = (depth_norm * 255.0).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2), "grayscale"

    raise ValueError("colormap must be inferno or grayscale")


def _encode_png(image):
    """Encode a Pillow image as PNG base64 without a data-URL prefix."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/health")
def health():
    """Report readiness, model ID, active device and the default depth colormap."""
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "device": DEVICE,
        "colormap_default": COLORMAP_DEFAULT,
    }


@app.post("/v1/depth")
async def depth(
    image: UploadFile = File(...),
    colormap: str = Form(COLORMAP_DEFAULT),
):
    """Estimate metric depth for one uploaded image.

    Args:
        image: Multipart image upload decoded as RGB.
        colormap: inferno or grayscale for the visualization.

    Returns:
        JSON-compatible depth metadata, float32 bytes in depth_meters_b64, PNG
        visualization, camera matrices and [height, width] shape/size.

    Raises:
        HTTPException: Empty/invalid image or unsupported colormap (HTTP 400).
    """
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="image payload is empty")
    try:
        pil_image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid image: {exc}"
        ) from exc

    process_res = max(pil_image.size)  # longer side (width, height)
    prediction = MODEL.inference([pil_image], process_res=process_res)
    depth_map = prediction.depth[0]  # [H, W] float32

    if not isinstance(depth_map, np.ndarray):
        depth_map = depth_map.detach().cpu().numpy()
    depth_map = depth_map.astype(np.float32)

    height, width = depth_map.shape
    depth_norm = _normalize_depth(depth_map)

    # Encode raw metric depth (meters) for programmatic use
    depth_meters_b64 = base64.b64encode(depth_map.tobytes()).decode("utf-8")

    # Colorized visualization
    try:
        colored, colormap_used = _apply_colormap(depth_norm, colormap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    depth_image = Image.fromarray(colored)
    depth_map_base64 = _encode_png(depth_image)

    return {
        "depth_meters_b64": depth_meters_b64,
        "depth_shape": [height, width],
        "depth_map_base64": depth_map_base64,
        "intrinsics": prediction.intrinsics.astype(float).tolist(),
        "extrinsics": prediction.extrinsics.astype(float).tolist(),
        "colormap": colormap_used,
        "model_id": MODEL_ID,
        "device": DEVICE,
        "size": [height, width],
    }


@app.post("/v1/pose")
async def pose(images: List[UploadFile] = File(...)):
    """Estimate camera matrices for two or more ordered image uploads.

    Returns intrinsics, world-to-camera extrinsics, image count and source sizes
    in [height, width] order. Empty, invalid or insufficient uploads raise an
    HTTPException with status 400.
    """
    if not images or len(images) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two images are required for camera pose estimation",
        )

    pil_images = []
    image_sizes = []
    for upload in images:
        payload = await upload.read()
        if not payload:
            raise HTTPException(
                status_code=400, detail=f"image payload is empty: {upload.filename}"
            )
        try:
            pil_image = Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"invalid image {upload.filename}: {exc}",
            ) from exc
        pil_images.append(pil_image)
        image_sizes.append([pil_image.size[1], pil_image.size[0]])  # [H, W]

    process_res = max(max(img.size) for img in pil_images)
    prediction = MODEL.inference(pil_images, process_res=process_res)

    return {
        "intrinsics": prediction.intrinsics.astype(float).tolist(),
        "extrinsics": prediction.extrinsics.astype(float).tolist(),
        "num_images": len(pil_images),
        "image_sizes": image_sizes,
        "model_id": MODEL_ID,
        "device": DEVICE,
    }


def main():
    """Serve the loaded DA3 model at the configured depth-service host and port."""
    import uvicorn

    host = os.getenv("DEPTH_ESTIMATION_HOST", "127.0.0.1")
    port = int(os.getenv("DEPTH_ESTIMATION_PORT", "8087"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
