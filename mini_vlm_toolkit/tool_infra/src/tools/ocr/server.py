import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = os.getenv("OCR_MODEL_ID", "PaddlePaddle/PaddleOCR-VL-1.5")
DEVICE_ENV = os.getenv("OCR_DEVICE", "auto")
TORCH_DTYPE_ENV = os.getenv("OCR_TORCH_DTYPE", "bfloat16")
MAX_NEW_TOKENS = int(os.getenv("OCR_MAX_NEW_TOKENS", "2048"))
REPETITION_PENALTY = float(os.getenv("OCR_REPETITION_PENALTY", "1.0"))

PROMPT = "OCR:"
MAX_PIXELS = 1280 * 28 * 28

app = FastAPI(title="OCR Server (PaddleOCR-VL)", version="0.4")


def _resolve_device(device):
    if device and device != "auto":
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(name):
    dtype = getattr(torch, str(name).strip(), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"OCR_TORCH_DTYPE={name!r} is not a valid torch dtype "
            "(expected e.g. bfloat16, float16, float32)"
        )
    return dtype


def _load_model():
    device = _resolve_device(DEVICE_ENV)
    dtype = _resolve_dtype(TORCH_DTYPE_ENV)
    model = (
        AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation="flash_attention_2")
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor, device


MODEL, PROCESSOR, DEVICE = _load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_id": MODEL_ID, "device": DEVICE}


@app.post("/v1/ocr")
async def ocr(image: UploadFile = File(...)):
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="image payload is empty")
    try:
        pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(payload))).convert(
            "RGB"
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    inputs = PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": PROCESSOR.image_processor.size["shortest_edge"],
                "longest_edge": MAX_PIXELS,
            }
        },
    ).to(DEVICE)

    gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS}
    if REPETITION_PENALTY != 1.0:
        gen_kwargs["repetition_penalty"] = REPETITION_PENALTY
    with torch.inference_mode():
        outputs = MODEL.generate(**inputs, **gen_kwargs)
    decoded = PROCESSOR.decode(outputs[0][inputs["input_ids"].shape[-1] : -1])

    return {"text": decoded.strip(), "model_id": MODEL_ID, "device": DEVICE}


def main():
    import uvicorn

    host = os.getenv("OCR_HOST", "127.0.0.1")
    port = int(os.getenv("OCR_PORT", "8085"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
