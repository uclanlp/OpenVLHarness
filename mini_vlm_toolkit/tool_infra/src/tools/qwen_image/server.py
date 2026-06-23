"""Qwen-Image diffusion text-to-image server.

Wraps Qwen/Qwen-Image (Apache 2.0) behind a FastAPI HTTP endpoint so the
main tool_infra server can call it as a heavy tool, parallel to SAM3 and
Depth Estimation. The model is loaded once at process start.

Endpoint:
    POST /v1/generate   JSON: {prompt, negative_prompt?, aspect_ratio?, seed?}
                        →    {image_base64, width, height, aspect_ratio,
                               seed_used, model_id, device}
    GET  /health        →    {status, model_id, device, aspect_ratio_default}

The ``image_base64`` key matches one of the framework's
``_IMAGE_RESULT_KEY_HINTS`` (see memory/modules/image.py), so the result
image is automatically registered as ``tool_generated_image_N`` and
attached to the agent's deferred user message.
"""

import base64
import io
import os
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

MODEL_ID = os.getenv("QWEN_IMAGE_MODEL_ID", "Qwen/Qwen-Image")
DEVICE_ENV = os.getenv("QWEN_IMAGE_DEVICE", "auto")
ASPECT_RATIO_DEFAULT = os.getenv("QWEN_IMAGE_ASPECT_RATIO", "1:1")
TORCH_DTYPE_ENV = os.getenv("QWEN_IMAGE_TORCH_DTYPE", "bfloat16")
NUM_INFERENCE_STEPS = int(os.getenv("QWEN_IMAGE_NUM_INFERENCE_STEPS", "50"))
GUIDANCE_SCALE = float(os.getenv("QWEN_IMAGE_GUIDANCE_SCALE", "4.0"))

# Qwen-Image's recommended pixel size per aspect ratio (from the model
# card). Keep this mapping centralized server-side so the tool stays
# stateless and the LLM-facing schema only carries the label.
ASPECT_RATIO_TO_SIZE = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

app = FastAPI(title="Qwen-Image Server", version="0.1")


def _resolve_device(device: str) -> str:
    if device and device != "auto":
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_torch_dtype(name: str):
    name = (name or "bfloat16").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {name}")


def _load_pipeline():
    from diffusers import DiffusionPipeline

    device = _resolve_device(DEVICE_ENV)
    dtype = _resolve_torch_dtype(TORCH_DTYPE_ENV)
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe = pipe.to(device)
    return pipe, device


PIPE, DEVICE = _load_pipeline()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = None
    aspect_ratio: Optional[str] = None
    seed: Optional[int] = None


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _resolve_seed(seed: Optional[int]) -> int:
    """Map ``None`` / ``-1`` to a fresh random seed; otherwise pass through.

    Returning the actually-used seed lets the caller reproduce the result
    later (Rui values reproducibility for benchmark debugging).
    """
    if seed is None or seed < 0:
        return int(torch.randint(0, 2**31 - 1, (1,)).item())
    return int(seed)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "device": DEVICE,
        "aspect_ratio_default": ASPECT_RATIO_DEFAULT,
    }


@app.post("/v1/generate")
def generate(request: GenerateRequest):
    aspect_ratio = request.aspect_ratio or ASPECT_RATIO_DEFAULT
    if aspect_ratio not in ASPECT_RATIO_TO_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"aspect_ratio must be one of "
                f"{sorted(ASPECT_RATIO_TO_SIZE.keys())}, got: {aspect_ratio!r}"
            ),
        )
    width, height = ASPECT_RATIO_TO_SIZE[aspect_ratio]
    seed_used = _resolve_seed(request.seed)
    generator = torch.Generator(device=DEVICE).manual_seed(seed_used)

    try:
        result = PIPE(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"inference failed: {exc}"
        ) from exc

    image = result.images[0]
    image_base64 = _encode_png(image)

    return {
        "image_base64": image_base64,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "seed_used": seed_used,
        "model_id": MODEL_ID,
        "device": DEVICE,
    }


def main():
    import uvicorn

    host = os.getenv("QWEN_IMAGE_HOST", "127.0.0.1")
    port = int(os.getenv("QWEN_IMAGE_PORT", "8088"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
