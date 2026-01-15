import io
import json
import os
import re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import easyocr
import numpy as np
from PIL import Image
import torch


LANGS_ENV = os.getenv("OCR_LANGS", "ch_sim,en")
GPU_ENV = os.getenv("OCR_GPU", "auto")
MODEL_DIR = os.getenv("OCR_MODEL_DIR", "").strip()
DOWNLOAD_ENABLED = os.getenv("OCR_DOWNLOAD_ENABLED", "true")

app = FastAPI()


def _parse_langs(raw):
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    parts = re.split(r"[,\s;|]+", raw)
    return [part.strip() for part in parts if part.strip()]


def _resolve_gpu():
    raw = str(GPU_ENV).lower().strip()
    if raw in ("1", "true", "yes", "on"):
        return torch.cuda.is_available()
    if raw in ("0", "false", "no", "off"):
        return False
    return torch.cuda.is_available()


def _resolve_download_enabled():
    return str(DOWNLOAD_ENABLED).lower() not in ("0", "false", "no", "off")


def _load_reader():
    langs = _parse_langs(LANGS_ENV)
    if not langs:
        langs = ["en"]
    gpu = _resolve_gpu()
    kwargs = {
        "gpu": gpu,
        "download_enabled": _resolve_download_enabled(),
    }
    if MODEL_DIR:
        kwargs["model_storage_directory"] = MODEL_DIR
    reader = easyocr.Reader(langs, **kwargs)
    return reader, langs, gpu


READER, LOADED_LANGS, USING_GPU = _load_reader()


@app.get("/health")
def health():
    return {"status": "ok", "languages": LOADED_LANGS, "gpu": USING_GPU}


@app.post("/v1/ocr")
async def ocr(
    image: UploadFile = File(...),
    languages: str = Form(""),
):
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="image payload is empty")

    requested_langs = _parse_langs(languages)
    if requested_langs and set(requested_langs) != set(LOADED_LANGS):
        raise HTTPException(
            status_code=400,
            detail=(
                "Requested languages do not match server configuration. "
                "Restart the OCR server with OCR_LANGS set to the desired list."
            ),
        )
    langs = LOADED_LANGS
    pil_image = Image.open(io.BytesIO(payload)).convert("RGB")
    results = READER.readtext(np.array(pil_image))

    texts = []
    structured = []
    for item in results:
        if len(item) < 3:
            continue
        box, text, confidence = item
        texts.append(text)
        structured.append(
            {
                "box": [[float(x), float(y)] for x, y in box],
                "text": text,
                "confidence": float(confidence),
            }
        )

    return {
        "texts": texts,
        "results": structured,
        "languages": langs,
        "gpu": USING_GPU,
    }


def main():
    import uvicorn

    host = os.getenv("OCR_HOST", "127.0.0.1")
    port = int(os.getenv("OCR_PORT", "8085"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
