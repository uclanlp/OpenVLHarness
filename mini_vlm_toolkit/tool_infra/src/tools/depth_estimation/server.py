import base64
import io
import os
import tempfile

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image


MODEL_ID = os.getenv(
    "DEPTH_ESTIMATION_MODEL_ID", "depth-anything/DA3NESTED-GIANT-LARGE"
)
DEVICE_ENV = os.getenv("DEPTH_ESTIMATION_DEVICE", "auto")
COLORMAP_DEFAULT = os.getenv("DEPTH_ESTIMATION_COLORMAP", "inferno")

app = FastAPI(title="Depth Estimation Server (DA3)", version="0.1")


def _resolve_device(device):
    if device and device != "auto":
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    from depth_anything_3.api import DepthAnything3

    device = _resolve_device(DEVICE_ENV)
    model = DepthAnything3.from_pretrained(MODEL_ID)
    model = model.to(device)
    return model, device


MODEL, DEVICE = _load_model()


def _normalize_depth(depth):
    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    if depth_max - depth_min < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return (depth - depth_min) / (depth_max - depth_min)


def _apply_colormap(depth_norm, colormap_name):
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
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/health")
def health():
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
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="image payload is empty")
    try:
        pil_image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid image: {exc}"
        ) from exc

    prediction = MODEL.inference([pil_image])
    depth_map = prediction.depth[0]  # [H, W] float32

    if not isinstance(depth_map, np.ndarray):
        depth_map = depth_map.detach().cpu().numpy()
    depth_map = depth_map.astype(np.float32)

    height, width = depth_map.shape
    depth_norm = _normalize_depth(depth_map)

    # Encode raw float32 depth for programmatic use
    depth_raw_b64 = base64.b64encode(depth_norm.tobytes()).decode("utf-8")

    # Colorized visualization
    try:
        colored, colormap_used = _apply_colormap(depth_norm, colormap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    depth_image = Image.fromarray(colored)
    depth_map_base64 = _encode_png(depth_image)

    return {
        "depth_raw_b64": depth_raw_b64,
        "depth_shape": [height, width],
        "depth_map_base64": depth_map_base64,
        "colormap": colormap_used,
        "model_id": MODEL_ID,
        "device": DEVICE,
        "size": [height, width],
    }


def main():
    import uvicorn

    host = os.getenv("DEPTH_ESTIMATION_HOST", "127.0.0.1")
    port = int(os.getenv("DEPTH_ESTIMATION_PORT", "8087"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
