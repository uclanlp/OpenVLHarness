"""Qwen-Image-Edit-2509 diffusion image-editing server.

Wraps Qwen/Qwen-Image-Edit-2509 (Apache 2.0) behind a FastAPI HTTP
endpoint so the main tool_infra server can call it as a heavy tool,
parallel to SAM3 / DA3 / Qwen-Image. The model is loaded once at
process start.

Endpoint:
    POST /v1/edit       JSON: {edit_prompt, image_base64s, negative_prompt?,
                               seed?, dump_dir?}
                        →    {image_base64, width, height, seed_used,
                              model_id, device}
    GET  /health        →    {status, model_id, device, edit_pipeline}

The ``image_base64`` key in the response matches one of the framework's
``_IMAGE_RESULT_KEY_HINTS`` (see memory/modules/image.py), so the result
image is automatically registered as ``tool_generated_image_N`` and
attached to the agent's deferred user message.
"""

import base64
import io
import os
import threading
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

# diffusers pipeline is not thread-safe; serialize concurrent /v1/edit requests
# so they don't corrupt the shared PIPE state ("index N out of bounds" 500s
# under NUM_PROC>1). See personal/log.md 2026-07-10 qwen-image concurrency note.
_PIPE_LOCK = threading.Lock()

MODEL_ID = os.getenv("QWEN_IMAGE_EDIT_MODEL_ID", "Qwen/Qwen-Image-Edit-2509")
DEVICE_ENV = os.getenv("QWEN_IMAGE_EDIT_DEVICE", "auto")
TORCH_DTYPE_ENV = os.getenv("QWEN_IMAGE_EDIT_TORCH_DTYPE", "bfloat16")
# Default Qwen-Image-Edit hyperparams (matching the upstream
# api.py / model-card recommendation).
NUM_INFERENCE_STEPS = int(os.getenv("QWEN_IMAGE_EDIT_NUM_INFERENCE_STEPS", "40"))
GUIDANCE_SCALE = float(os.getenv("QWEN_IMAGE_EDIT_GUIDANCE_SCALE", "1.0"))
TRUE_CFG_SCALE = float(os.getenv("QWEN_IMAGE_EDIT_TRUE_CFG_SCALE", "4.0"))
# Negative prompt: upstream Qwen recipe uses a single space rather than
# an empty string. Empty string degrades quality on some diffusers
# pipelines; the single space is a known-good no-op anchor.
NEGATIVE_PROMPT_DEFAULT = os.getenv("QWEN_IMAGE_EDIT_NEGATIVE_PROMPT", " ")
# Loading mode: same contract as the qwen_image tool. Qwen-Image-Edit
# is smaller than Qwen-Image; Gen-Searcher reference deployment fits
# one instance per single GPU (no sharding needed). We keep the same
# LOAD_MODE switches so the operator can opt into balanced / offload
# if their hardware demands it.
#   "single"   — whole pipeline on a single device (recommended for
#                single-GPU deploy; should fit on a 48 GB A6000).
#   "balanced" — diffusers/accelerate device_map="balanced" sharding
#                across all visible GPUs (use only if single-GPU OOMs).
#   "offload"  — pipe.enable_model_cpu_offload(); 2-3× slower.
#   "sequential" — pipe.enable_sequential_cpu_offload(); 5-10× slower.
LOAD_MODE = os.getenv("QWEN_IMAGE_EDIT_LOAD_MODE", "single").lower()
# Optional: if set, dump every generated PNG to this directory in
# addition to returning it base64 in the response body. Useful for
# offline inspection. Unset by default to keep the server stateless.
DUMP_DIR = os.getenv("QWEN_IMAGE_EDIT_DUMP_DIR", "").strip()

app = FastAPI(title="Qwen-Image-Edit Server", version="0.1")


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
    from diffusers import QwenImageEditPlusPipeline

    device = _resolve_device(DEVICE_ENV)
    dtype = _resolve_torch_dtype(TORCH_DTYPE_ENV)

    if LOAD_MODE == "balanced":
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID, torch_dtype=dtype, device_map="balanced",
            use_safetensors=True,
        )
        return pipe, "cuda:balanced"

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, use_safetensors=True,
    )

    if LOAD_MODE == "offload":
        pipe.enable_model_cpu_offload()
        return pipe, "cuda:cpu_offload"
    if LOAD_MODE == "sequential":
        pipe.enable_sequential_cpu_offload()
        return pipe, "cuda:sequential_cpu_offload"
    if LOAD_MODE != "single":
        raise ValueError(
            f"unsupported QWEN_IMAGE_EDIT_LOAD_MODE={LOAD_MODE!r}; "
            "expected single / balanced / offload / sequential"
        )

    pipe = pipe.to(device)
    return pipe, device


PIPE, DEVICE = _load_pipeline()


class EditRequest(BaseModel):
    edit_prompt: str = Field(..., min_length=1)
    image_base64s: List[str] = Field(..., min_length=1)
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    # Per-request dump-dir override. When set (non-empty absolute path),
    # this request's PNG is written here instead of the server-startup
    # default (QWEN_IMAGE_EDIT_DUMP_DIR). When unset, the server-startup
    # default is used (which may itself be unset → no dump). Intentionally
    # NOT exposed in the tool-side INPUT_SCHEMA — this is an ops/operator
    # parameter, never an agent decision.
    dump_dir: Optional[str] = None


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _decode_data_url_or_bare_b64(value: str) -> Image.Image:
    """Accept either ``data:image/...;base64,<b64>`` (the form the
    framework's ImageModule emits) or a bare base64 string (manual
    callers). Returns an RGB PIL Image.
    """
    raw = value.strip()
    if raw.startswith("data:image/"):
        # Strip the ``data:image/<mime>;base64,`` prefix.
        try:
            _, payload = raw.split(",", 1)
        except ValueError as exc:
            raise ValueError("malformed data URL: missing comma") from exc
        b64 = payload
    else:
        b64 = raw
    try:
        image_bytes = base64.b64decode(b64, validate=False)
    except Exception as exc:
        raise ValueError(f"image_base64 not decodable: {exc}") from exc
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"image bytes not a valid image: {exc}") from exc


def _resolve_seed(seed: Optional[int]) -> int:
    """Map ``None`` / ``-1`` to a fresh random seed; otherwise pass through."""
    if seed is None or seed < 0:
        return int(torch.randint(0, 2**31 - 1, (1,)).item())
    return int(seed)


def _maybe_dump_png(
    image: Image.Image,
    seed_used: int,
    n_inputs: int,
    dump_dir_override: Optional[str] = None,
) -> Optional[str]:
    """Best-effort write a copy of the generated PNG. Priority:
    per-request override > server-startup env > no dump.
    """
    target = (dump_dir_override or DUMP_DIR or "").strip()
    if not target:
        return None
    try:
        os.makedirs(target, exist_ok=True)
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{seed_used}_{n_inputs}in.png"
        path = os.path.join(target, filename)
        image.save(path, format="PNG")
        return path
    except Exception as exc:
        print(f"[qwen_image_edit] warn: failed to dump PNG to {target}: {exc}")
        return None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "device": DEVICE,
        # Pipeline-family fingerprint for scan_server_status to recognize
        # this server vs the sibling Qwen-Image text-to-image one. The
        # presence of ``edit_pipeline`` is the distinguishing field.
        "edit_pipeline": "QwenImageEditPlusPipeline",
    }


@app.post("/v1/edit")
def edit(request: EditRequest):
    # Decode all input images first so a single malformed payload
    # short-circuits with a clear 400 rather than failing inside the
    # diffusion pipeline.
    try:
        pil_images = [_decode_data_url_or_bare_b64(b) for b in request.image_base64s]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    negative = request.negative_prompt
    if negative is None:
        negative = NEGATIVE_PROMPT_DEFAULT

    seed_used = _resolve_seed(request.seed)
    # Same convention as the Qwen-Image text-to-image server: pin the
    # RNG to cuda:0 (or cpu) — torch.Generator does not accept the
    # report-only "cuda:balanced" / "cuda:cpu_offload" labels.
    gen_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed_used)

    try:
        with _PIPE_LOCK, torch.inference_mode():
            result = PIPE(
                image=pil_images,
                prompt=request.edit_prompt,
                negative_prompt=negative,
                generator=generator,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                true_cfg_scale=TRUE_CFG_SCALE,
                num_images_per_prompt=1,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"inference failed: {exc}"
        ) from exc

    image = result.images[0]
    image_base64 = _encode_png(image)
    dumped_path = _maybe_dump_png(
        image, seed_used, len(pil_images),
        dump_dir_override=request.dump_dir,
    )

    response = {
        "image_base64": image_base64,
        "width": image.size[0],
        "height": image.size[1],
        "input_image_count": len(pil_images),
        "seed_used": seed_used,
        "model_id": MODEL_ID,
        "device": DEVICE,
    }
    if dumped_path:
        response["dumped_to"] = dumped_path
    return response


def main():
    import uvicorn

    host = os.getenv("QWEN_IMAGE_EDIT_HOST", "127.0.0.1")
    port = int(os.getenv("QWEN_IMAGE_EDIT_PORT", "8090"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
