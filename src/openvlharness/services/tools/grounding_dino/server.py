import io
import json
import os
import re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


MODEL_ID = os.getenv("GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-base")
DEVICE_ENV = os.getenv("GROUNDING_DINO_DEVICE", "auto")
DEFAULT_BOX_THRESHOLD = float(os.getenv("GROUNDING_DINO_BOX_THRESHOLD", "0.4"))
DEFAULT_TEXT_THRESHOLD = float(os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", "0.3"))

app = FastAPI()


def _resolve_device(device):
    if device and device != "auto":
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    device = _resolve_device(DEVICE_ENV)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return model, processor, device


MODEL, PROCESSOR, DEVICE = _load_model()


def _parse_labels(query):
    raw = query.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    parts = re.split(r"[.,;|]+", raw)
    return [part.strip() for part in parts if part.strip()]


def _build_inputs(processor, image, labels, device):
    try:
        return processor(images=image, text=[labels], return_tensors="pt").to(device)
    except TypeError:
        prompt = " . ".join(labels).strip()
        if prompt and not prompt.endswith("."):
            prompt = f"{prompt} ."
        return processor(images=image, text=prompt, return_tensors="pt").to(device)


def _normalize_boxes(boxes, width, height):
    # Boxes are returned on a 0-1000 scale (Qwen-style) with 2-decimal precision.
    normalized = []
    if not width or not height:
        return normalized
    for box in boxes:
        x1, y1, x2, y2 = [float(value) for value in box]
        normalized.append(
            [
                round(x1 / width * 1000.0, 2),
                round(y1 / height * 1000.0, 2),
                round(x2 / width * 1000.0, 2),
                round(y2 / height * 1000.0, 2),
            ]
        )
    return normalized


def _coerce_labels(raw_labels, label_pool):
    labels = []
    for label in raw_labels:
        if isinstance(label, (int, float)):
            idx = int(label)
            if 0 <= idx < len(label_pool):
                labels.append(label_pool[idx])
            else:
                labels.append(str(label))
        else:
            labels.append(str(label))
    return labels


@app.get("/health")
def health():
    return {"status": "ok", "model_id": MODEL_ID, "device": DEVICE}


@app.post("/v1/grounding_dino")
async def grounding_dino(
    image: UploadFile = File(...),
    query: str = Form(...),
    box_threshold: float = Form(DEFAULT_BOX_THRESHOLD),
    text_threshold: float = Form(DEFAULT_TEXT_THRESHOLD),
):
    labels = _parse_labels(query)
    if not labels:
        raise HTTPException(status_code=400, detail="query must include at least one label")

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="image payload is empty")
    pil_image = Image.open(io.BytesIO(payload)).convert("RGB")
    width, height = pil_image.size

    inputs = _build_inputs(PROCESSOR, pil_image, labels, DEVICE)
    with torch.no_grad():
        outputs = MODEL(**inputs)

    try:
        results = PROCESSOR.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=[(height, width)],
        )
    except TypeError:
        results = PROCESSOR.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=[(height, width)],
        )
    result = results[0] if results else {}

    boxes_tensor = result.get("boxes")
    scores_tensor = result.get("scores")
    raw_labels = result.get("labels", [])

    boxes = (
        [[round(float(value), 2) for value in row] for row in boxes_tensor.tolist()]
        if boxes_tensor is not None
        else []
    )
    scores = (
        [round(float(value), 3) for value in scores_tensor.tolist()]
        if scores_tensor is not None
        else []
    )
    resolved_labels = _coerce_labels(raw_labels, labels)
    boxes_normalized = _normalize_boxes(boxes, width, height)

    return {
        "boxes": boxes,
        "boxes_normalized": boxes_normalized,
        "scores": scores,
        "labels": resolved_labels,
        "size": [height, width],
        "model_id": MODEL_ID,
    }


def main():
    import uvicorn

    host = os.getenv("GROUNDING_DINO_HOST", "127.0.0.1")
    port = int(os.getenv("GROUNDING_DINO_PORT", "8084"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
