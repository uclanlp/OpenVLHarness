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
import threading
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

# The diffusers pipeline (global PIPE) is NOT thread-safe: FastAPI runs sync
# endpoints in a threadpool, so concurrent /v1/generate requests (e.g. an eval
# with NUM_PROC>1) hit the same pipeline object at once and corrupt its shared
# tensor state -> "index N out of bounds" 500s. Serialize inference on this
# lock so requests queue and run one-at-a-time on the GPU.
_PIPE_LOCK = threading.Lock()

MODEL_ID = os.getenv("QWEN_IMAGE_MODEL_ID", "Qwen/Qwen-Image")
DEVICE_ENV = os.getenv("QWEN_IMAGE_DEVICE", "auto")
ASPECT_RATIO_DEFAULT = os.getenv("QWEN_IMAGE_ASPECT_RATIO", "1:1")
TORCH_DTYPE_ENV = os.getenv("QWEN_IMAGE_TORCH_DTYPE", "bfloat16")
NUM_INFERENCE_STEPS = int(os.getenv("QWEN_IMAGE_NUM_INFERENCE_STEPS", "50"))
GUIDANCE_SCALE = float(os.getenv("QWEN_IMAGE_GUIDANCE_SCALE", "4.0"))
# Loading mode:
#   "single"   — load whole pipeline onto a single device (default; needs
#                ~40 GB free VRAM, which a single 48 GB A6000 cannot
#                actually sustain alongside activations → OOM).
#   "balanced" — diffusers/accelerate device_map="balanced" sharding
#                across all visible GPUs. Use when CUDA_VISIBLE_DEVICES
#                has 2+ GPUs (e.g. "5,6"); inference stays near
#                single-card speed because the model splits cleanly
#                across the cards.
#   "offload"  — pipe.enable_model_cpu_offload(); weights live on CPU
#                and per-layer swap to GPU. Single-GPU friendly but
#                2-3× slower per step.
#   "sequential" — pipe.enable_sequential_cpu_offload(); maximum VRAM
#                  savings, 5-10× slower; only as last resort.
LOAD_MODE = os.getenv("QWEN_IMAGE_LOAD_MODE", "single").lower()
# Optional: if set, dump every generated PNG to this directory in addition
# to returning it base64 in the response body. Useful for offline
# inspection of agent runs. Unset by default to keep the server stateless.
DUMP_DIR = os.getenv("QWEN_IMAGE_DUMP_DIR", "").strip()

# Qwen-Image's recommended pixel size per aspect ratio (from the model
# card). Keep this mapping centralized server-side so the tool stays
# stateless and the LLM-facing schema only carries the label.
#
# NOTE: diffusers' VAE silently floor()s width/height to the nearest
# multiple of 8. The 4:3 / 3:4 sizes in the model card (1472x1140) are
# NOT multiples of 8 (1140/8=142.5), so the actual generated image is
# 1472x1136 not 1472x1140. We use the aligned numbers here so the
# response width/height field matches the bytes the caller actually
# receives. Verified live 2026-06-23 via end-to-end test T4.
ASPECT_RATIO_TO_SIZE = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1136),
    "3:4": (1136, 1472),
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

    if LOAD_MODE == "balanced":
        # Sharding requires diffusers to lay out the modules itself; we
        # must NOT call .to(device) afterwards. Reported device is the
        # logical label for /health; the actual placement spans every
        # GPU listed in CUDA_VISIBLE_DEVICES.
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=dtype, device_map="balanced",
        )
        return pipe, "cuda:balanced"

    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)

    if LOAD_MODE == "offload":
        pipe.enable_model_cpu_offload()
        return pipe, "cuda:cpu_offload"
    if LOAD_MODE == "sequential":
        pipe.enable_sequential_cpu_offload()
        return pipe, "cuda:sequential_cpu_offload"
    if LOAD_MODE != "single":
        raise ValueError(
            f"unsupported QWEN_IMAGE_LOAD_MODE={LOAD_MODE!r}; "
            "expected single / balanced / offload / sequential"
        )

    pipe = pipe.to(device)
    return pipe, device


PIPE, DEVICE = _load_pipeline()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = None
    aspect_ratio: Optional[str] = None
    seed: Optional[int] = None
    # Per-request dump-dir override. When set (non-empty absolute path),
    # this request's PNG is written here instead of the server-startup
    # default (QWEN_IMAGE_DUMP_DIR). When unset, the server-startup
    # default is used (which may itself be unset → no dump). Intentionally
    # NOT exposed in the tool-side INPUT_SCHEMA — this is an ops/operator
    # parameter, never an agent decision.
    dump_dir: Optional[str] = None


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _maybe_dump_png(
    image: Image.Image,
    seed_used: int,
    aspect_ratio: str,
    dump_dir_override: Optional[str] = None,
) -> Optional[str]:
    """Best-effort write a copy of the generated PNG.

    Path priority: per-request ``dump_dir_override`` > server-startup
    ``DUMP_DIR`` env > no dump. Filename pattern:
    ``YYYYmmdd_HHMMSS_<seed>_<aspect>.png`` so the same seed across runs
    is easy to spot. Returns the path on success, ``None`` if dumping is
    disabled or fails (failures are logged but never break the response —
    the base64 payload is the authoritative result).
    """
    target = (dump_dir_override or DUMP_DIR or "").strip()
    if not target:
        return None
    try:
        os.makedirs(target, exist_ok=True)
        # Local import keeps the smoke-test stub light (datetime isn't
        # in the offline test path).
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_aspect = aspect_ratio.replace(":", "x")
        filename = f"{ts}_{seed_used}_{safe_aspect}.png"
        path = os.path.join(target, filename)
        image.save(path, format="PNG")
        return path
    except Exception as exc:
        # Don't ever take the generation down because of a disk hiccup.
        # Surface the error in the server log; client still gets the PNG
        # in the response body.
        print(f"[qwen_image] warn: failed to dump PNG to {target}: {exc}")
        return None


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
    # torch.Generator needs a concrete device string (not the report-only
    # "cuda:balanced" / "cuda:cpu_offload" labels we return from /health).
    # Pin the RNG to cuda:0 (or cpu when CUDA is absent); diffusion noise
    # tensors get moved by the pipeline anyway.
    gen_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed_used)

    try:
        with _PIPE_LOCK, torch.inference_mode():
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
    dumped_path = _maybe_dump_png(
        image, seed_used, aspect_ratio,
        dump_dir_override=request.dump_dir,
    )

    # Report the *actual* generated size (diffusers may have silently
    # floor()'d width/height to a multiple of 8 for VAE alignment). The
    # request-intended (width, height) is recorded separately as
    # ``requested_size`` for debugging; agents and downstream code
    # should always use ``width``/``height``.
    actual_w, actual_h = image.size
    response = {
        "image_base64": image_base64,
        "width": actual_w,
        "height": actual_h,
        "requested_size": [width, height],
        "aspect_ratio": aspect_ratio,
        "seed_used": seed_used,
        "model_id": MODEL_ID,
        "device": DEVICE,
    }
    # Only surface the path when dumping is on. Kept out of the base
    # response so framework consumers (ImageModule etc.) don't see an
    # unexpected extra field they have to filter.
    if dumped_path:
        response["dumped_to"] = dumped_path
    return response


def main():
    import uvicorn

    host = os.getenv("QWEN_IMAGE_HOST", "127.0.0.1")
    port = int(os.getenv("QWEN_IMAGE_PORT", "8088"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
