import base64
import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_ID = os.getenv("DEPTH_ANYTHING_MODEL_ID", "LiheYoung/depth-anything-base-hf")
DEVICE_ENV = os.getenv("DEPTH_ANYTHING_DEVICE", "cpu")
COLORMAP_DEFAULT = os.getenv("DEPTH_ANYTHING_COLORMAP", "inferno")

app = FastAPI()


def _resolve_device(device):
    if device and device != "auto":
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    device = _resolve_device(DEVICE_ENV)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return model, processor, device


MODEL, PROCESSOR, DEVICE = _load_model()


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
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    inputs = PROCESSOR(images=pil_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(**inputs)

    predicted_depth = outputs.predicted_depth
    if predicted_depth.dim() == 3:
        predicted_depth = predicted_depth.unsqueeze(1)
    if predicted_depth.dim() != 4:
        raise HTTPException(status_code=500, detail="unexpected depth output shape")

    width, height = pil_image.size
    resized = F.interpolate(
        predicted_depth,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    depth_map = resized.squeeze().detach().cpu().numpy().astype(np.float32)
    depth_norm = _normalize_depth(depth_map)

    try:
        colored, colormap_used = _apply_colormap(depth_norm, colormap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    depth_image = Image.fromarray(colored)
    depth_base64 = _encode_png(depth_image)

    return {
        "depth_map_base64": depth_base64,
        "colormap": colormap_used,
        "model_id": MODEL_ID,
        "device": DEVICE,
        "size": [height, width],
    }


def main():
    import uvicorn

    host = os.getenv("DEPTH_ANYTHING_HOST", "127.0.0.1")
    port = int(os.getenv("DEPTH_ANYTHING_PORT", "8082"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
