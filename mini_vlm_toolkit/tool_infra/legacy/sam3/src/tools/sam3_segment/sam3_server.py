import base64
import io
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

MODEL_NAME = "sam3-image"

app = FastAPI(title="SAM3 Segmentation Server", version="0.1")

# Load once at startup (important for performance)
model = build_sam3_image_model()
processor = Sam3Processor(model)


def _pil_from_upload(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if not data:
        raise ValueError("Empty image upload.")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img


def _mask_to_png_base64(mask: np.ndarray) -> str:
    """
    Convert a single mask to a PNG (binary/uint8) and base64-encode it.
    Expected mask shape: (H, W) or (1, H, W).
    """
    m = mask
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    m = (m > 0).astype(np.uint8) * 255  # 0/255
    pil = Image.fromarray(m, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/segment")
def segment(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    return_masks_png_b64: bool = Form(True),
    max_masks: Optional[int] = Form(None),
) -> JSONResponse:
    """
    Multipart/form-data request:
      - prompt: text prompt
      - image: uploaded image file
      - return_masks_png_b64: whether to return masks as PNG base64 strings
      - max_masks: optional limit on number of masks returned
    """
    try:
        pil_img = _pil_from_upload(image)

        inference_state = processor.set_image(pil_img)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        # Convert to numpy / python-native types
        masks_np = None
        if masks is not None:
            try:
                import torch  # optional
                if "torch" in str(type(masks)):
                    masks_np = masks.detach().cpu().numpy()
                else:
                    masks_np = np.array(masks)
            except Exception:
                masks_np = np.array(masks)

        boxes_list: List[Any] = []
        scores_list: List[float] = []

        if boxes is not None:
            try:
                import torch
                if "torch" in str(type(boxes)):
                    boxes_list = boxes.detach().cpu().tolist()
                else:
                    boxes_list = np.array(boxes).tolist()
            except Exception:
                boxes_list = np.array(boxes).tolist()

        if scores is not None:
            try:
                import torch
                if "torch" in str(type(scores)):
                    scores_list = scores.detach().cpu().tolist()
                else:
                    scores_list = np.array(scores).tolist()
            except Exception:
                scores_list = np.array(scores).tolist()

        # Apply optional truncation
        if max_masks is not None and masks_np is not None:
            masks_np = masks_np[:max_masks]
            boxes_list = boxes_list[:max_masks]
            scores_list = scores_list[:max_masks]

        masks_png_b64: List[str] = []
        if return_masks_png_b64 and masks_np is not None:
            # Common shapes: (N, H, W) or (N, 1, H, W)
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                for i in range(masks_np.shape[0]):
                    masks_png_b64.append(_mask_to_png_base64(masks_np[i]))
            elif masks_np.ndim == 3:
                for i in range(masks_np.shape[0]):
                    masks_png_b64.append(_mask_to_png_base64(masks_np[i]))
            else:
                masks_png_b64.append(_mask_to_png_base64(masks_np))

        num_masks = 0
        if masks_np is not None:
            if masks_np.ndim >= 3:
                num_masks = int(masks_np.shape[0])
            else:
                num_masks = 1

        resp: Dict[str, Any] = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "boxes": boxes_list,
            "scores": scores_list,
            "num_masks": num_masks,
        }
        if return_masks_png_b64:
            resp["masks_png_b64"] = masks_png_b64

        return JSONResponse(resp)

    except Exception as e:
        return JSONResponse({"model": MODEL_NAME, "error": str(e)}, status_code=400)
