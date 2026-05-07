"""Examine_Subimages_Tool -- horizontal-scaling per-subimage yes/no MLLM check.

For each box in `boxes` and each object in `objects`, send a yes/no visual
question to a fresh MLLM context (via the same OpenAI-compatible endpoint as
`Python_Coding_Agent_Tool`) and aggregate the answers. Object indices refer
to the most recent `Visual_Grounding_Tool` call on the same `image_id`; the
agent passes that grounding result inline as `_grounding_result` (a dict
with `pred_masks`, `pred_boxes`, `orig_img_h`, `orig_img_w`), so the tool
server doesn't need filesystem access to a per-task sidecar.

Configuration (env vars):
    CODING_AGENT_BASE_URL  - OpenAI-compatible endpoint (e.g. http://localhost:8004/v1)
    CODING_AGENT_API_KEY   - API key (default: "empty")
    CODING_AGENT_MODEL     - model name served by the endpoint
    EXAMINE_SUBIMAGES_MAX_WORKERS - thread pool size (default: 8)
"""

import base64
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from tools.api_utils import chat_completion_request
from tools.base import BaseTool
from tools.image_utils import coerce_xyxy
from tools.viz import visualize


TOOL_NAME = "Examine_Subimages_Tool"

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {
            "type": "string",
            "description": (
                "A runtime image ID from Environment Update messages, e.g. "
                "input_image_1 or tool_generated_image_1."
            ),
        },
        "image_base64": {
            "type": "string",
            "description": (
                "Internal: data URL for the source image. Populated "
                "automatically by the agent's image-id resolver from "
                "`image_id`. Do not set this directly."
            ),
        },
        "question": {
            "type": "string",
            "description": (
                "A clear yes/no question that can be answered by visual "
                "inspection of each subimage or object."
            ),
        },
        "objects": {
            "type": "array",
            "description": (
                "1-indexed object numbers from the most recent "
                "Visual_Grounding_Tool call on this image_id. Optional; "
                "either `objects` or `boxes` (or both) must be provided."
            ),
            "items": {"type": "integer", "minimum": 1},
        },
        "boxes": {
            "type": "array",
            "description": (
                "List of normalized [x1, y1, x2, y2] boxes on a 0-1000 scale "
                "(every value in [0, 1000], (x1, y1) top-left, (x2, y2) "
                "bottom-right). Optional; either `objects` or `boxes` (or "
                "both) must be provided."
            ),
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
        },
    },
    "required": ["image_id", "question"],
    "additionalProperties": False,
}

_PROMPTS_DIR = Path(__file__).parent
_OBJ_SYS_PROMPT = (_PROMPTS_DIR / "system_prompt_iterative_check.txt").read_text()
_BOX_SYS_PROMPT = (_PROMPTS_DIR / "system_prompt_box_iterative_check.txt").read_text()

_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_YES_TOKENS = {"Y", "YES", "TRUE", "CORRECT", "AFFIRMATIVE"}
_NO_TOKENS = {"N", "NO", "FALSE", "INCORRECT", "NEGATIVE"}


def _parse_yes_no(text):
    if not isinstance(text, str):
        return "NO"
    match = _TAG_RE.search(text)
    if match:
        candidate = match.group(1).strip().upper()
    else:
        upper = text.upper()
        if re.search(r"\bYES\b", upper):
            candidate = "YES"
        elif re.search(r"\bNO\b", upper):
            candidate = "NO"
        else:
            return "NO"
    if candidate in _YES_TOKENS:
        return "YES"
    if candidate in _NO_TOKENS:
        return "NO"
    return "NO"


def _decode_data_url(data_url):
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise ValueError("expected a data: URL for the image")
    header, _, b64 = data_url.partition(",")
    if not b64:
        raise ValueError("data URL missing base64 payload")
    return base64.b64decode(b64)


def _validate_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return False
    if any(v < 0 or v > 1000 for v in vals):
        return False
    if vals[0] >= vals[2] or vals[1] >= vals[3]:
        return False
    return True


_BBOX_REQ = (
    "Each box must be [x1, y1, x2, y2] with all four values in [0, 1000], "
    "x1 < x2, and y1 < y2."
)


class Examine_Subimages_Tool(BaseTool):
    require_llm_engine = False

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Send a yes/no visual question to a fresh-context MLLM for "
                "each provided subimage in parallel, and aggregate the "
                "answers. For each box in `boxes`, the tool zooms into that "
                "region (with default padding=50 on the 0-1000 scale) and "
                "asks the question. For each object number in `objects` "
                "(1-indexed, from the most recent Visual_Grounding_Tool "
                "call on this `image_id`), the tool renders the object's "
                "contrasting-colored mask context view + zoomed view and "
                "asks the same question. Use this to filter or count "
                "objects by attribute (e.g. 'people looking right', 'ripe "
                "red apples'). Either `objects` or `boxes` must be "
                "provided; both are allowed."
            ),
            tool_version="0.1.0",
            input_types={
                "image_id": "str - runtime image id, e.g. input_image_1",
                "question": "str - a clear yes/no visual question",
                "objects": (
                    "list[int] - 1-indexed object numbers from the most "
                    "recent Visual_Grounding_Tool call on image_id"
                ),
                "boxes": (
                    "list[list[float]] - normalized [x1, y1, x2, y2] boxes "
                    "on a 0-1000 scale (each value in [0, 1000])"
                ),
            },
            input_schema=INPUT_SCHEMA,
            output_type=(
                "dict - yes_count, no_count, yes_objects, no_objects, "
                "yes_boxes, no_boxes"
            ),
            demo_commands=[
                {
                    "command": (
                        'Examine_Subimages_Tool(image_id="input_image_1", '
                        'question="Is this person looking right?", '
                        "objects=[1, 2, 3, 4])"
                    ),
                    "description": (
                        "After Visual_Grounding_Tool grounded 4 people, "
                        "filter to those looking right."
                    ),
                },
                {
                    "command": (
                        'Examine_Subimages_Tool(image_id="input_image_1", '
                        'question="Is the apple ripe?", '
                        "boxes=[[120, 240, 380, 510], [600, 100, 820, 320]])"
                    ),
                    "description": (
                        "Ask the same question of two custom regions when "
                        "no grounding result is available."
                    ),
                },
            ],
        )
        self._base_url = os.getenv("CODING_AGENT_BASE_URL", "")
        self._base_url = (
            self._normalize_endpoint(self._base_url) if self._base_url else ""
        )
        self._api_key = os.getenv("CODING_AGENT_API_KEY", "empty")
        self._model = os.getenv("CODING_AGENT_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        self._max_workers = int(os.getenv("EXAMINE_SUBIMAGES_MAX_WORKERS", "8"))

    @staticmethod
    def _normalize_endpoint(base_url):
        trimmed = base_url.rstrip("/")
        if trimmed.endswith("/v1"):
            return f"{trimmed}/chat/completions"
        if trimmed.endswith("completions"):
            return trimmed
        return f"{trimmed}/v1/chat/completions"

    # No sanity_check: env-var validation happens lazily in execute() so
    # the tool still registers when CODING_AGENT_* are unset (mirrors the
    # Python_Coding_Agent_Tool pattern). In auto-discovery mode the loader
    # silently skips tools whose sanity_check returns issues, which would
    # hide this tool from /tools entirely.

    # ------------------------------------------------------------------ #
    # Image materialization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _materialize_image(image_data_url, tmp_paths):
        raw = _decode_data_url(image_data_url)
        # Detect format from the data URL header to pick a reasonable suffix.
        suffix = ".png"
        try:
            header = image_data_url.split(",", 1)[0]
            mime = header[5:].split(";", 1)[0].strip()
            if mime == "image/jpeg":
                suffix = ".jpg"
            elif mime == "image/webp":
                suffix = ".webp"
        except Exception:
            pass
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="examine_src_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            raise
        tmp_paths.append(path)
        return path

    @staticmethod
    def _save_pil_temp(pil_img, tmp_paths, prefix="examine_view_"):
        fd, path = tempfile.mkstemp(suffix=".png", prefix=prefix)
        os.close(fd)
        pil_img.save(path, format="PNG")
        tmp_paths.append(path)
        return path

    # ------------------------------------------------------------------ #
    # Single-query VLM call
    # ------------------------------------------------------------------ #

    def _query_one(self, system_prompt, question, image_paths):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        try:
            response = chat_completion_request(
                model=self._model,
                messages=messages,
                base_url=self._base_url,
                api_key=self._api_key,
                temperature=0.0,
                max_tokens=512,
                image_paths=image_paths,
            )
        except Exception:
            return "NO"
        content = ""
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                content = message.get("content") or ""
        return _parse_yes_no(content)

    # ------------------------------------------------------------------ #
    # Execute
    # ------------------------------------------------------------------ #

    def execute(
        self,
        image_id=None,
        question=None,
        objects=None,
        boxes=None,
        image_base64=None,
        _image_id=None,
        _grounding_result=None,
    ):
        # The agent's image-id resolver pops `image_id` and inserts
        # `image_base64` (a data URL) before this method runs. The agent
        # also injects:
        #   - `_image_id`: the original image_id (resolver pops the user-
        #     facing one), used in error messages.
        #   - `_grounding_result`: the latest Visual_Grounding_Tool result
        #     for that image_id, with keys (pred_masks, pred_boxes,
        #     orig_img_h, orig_img_w). Passed inline so the tool server
        #     never needs filesystem access to a sidecar.
        effective_image_id = (
            _image_id if isinstance(_image_id, str) and _image_id
            else image_id
        )
        if not effective_image_id or not isinstance(effective_image_id, str):
            return {"error": "image_id is required and must be a string"}
        if not question or not isinstance(question, str):
            return {"error": "question is required and must be a string"}
        if not self._base_url:
            return {"error": "CODING_AGENT_BASE_URL is not configured"}
        if not self._model:
            return {"error": "CODING_AGENT_MODEL is not configured"}

        objects = list(objects) if objects else []
        boxes_list = list(boxes) if boxes else []

        if not objects and not boxes_list:
            return {
                "error": (
                    "must provide at least one of `objects` or `boxes`"
                )
            }

        if boxes_list:
            for box in boxes_list:
                if not _validate_box(box):
                    return {
                        "error": (
                            f"Invalid bbox {box!r}. {_BBOX_REQ}"
                        )
                    }

        if not image_base64:
            return {
                "error": (
                    f"Could not resolve image data for image_id "
                    f"{effective_image_id!r}. Please verify the id appears "
                    "in the latest Environment Update message."
                )
            }

        detection = None
        if objects:
            if not isinstance(_grounding_result, dict):
                return {
                    "error": (
                        f"image {effective_image_id!r} has not been used as "
                        "the grounding target of the visual grounding tool, "
                        "please check the messages history and use the "
                        "right image"
                    )
                }
            detection = _grounding_result
            pred_masks = detection.get("pred_masks") or []
            n_masks = len(pred_masks)
            if n_masks == 0:
                return {
                    "error": (
                        f"latest Visual_Grounding_Tool call on "
                        f"{effective_image_id!r} produced 0 objects; "
                        "nothing to examine"
                    )
                }
            for obj_num in objects:
                if not isinstance(obj_num, int) or obj_num < 1 or obj_num > n_masks:
                    return {
                        "error": (
                            f"object {obj_num} exceeds the available "
                            f"object count ({n_masks}) for image "
                            f"{effective_image_id!r}"
                        )
                    }

        tmp_paths = []
        tasks = []  # list of (kind, key, system_prompt, image_paths)
        try:
            src_path = self._materialize_image(image_base64, tmp_paths)

            if boxes_list:
                pil_src = Image.open(src_path).convert("RGB")
                src_w, src_h = pil_src.size
                for box in boxes_list:
                    x1, y1, x2, y2 = [float(v) for v in box]
                    pad = 50.0
                    px1, py1, px2, py2 = (
                        max(0.0, x1 - pad),
                        max(0.0, y1 - pad),
                        min(1000.0, x2 + pad),
                        min(1000.0, y2 + pad),
                    )
                    left, top, right, bottom = coerce_xyxy(
                        [px1, py1, px2, py2], src_w, src_h, "normalized_xyxy"
                    )
                    crop = pil_src.crop((left, top, right, bottom))
                    crop_path = self._save_pil_temp(
                        crop, tmp_paths, prefix="examine_box_"
                    )
                    tasks.append(
                        ("box", tuple(box), _BOX_SYS_PROMPT, [crop_path])
                    )

            if objects:
                input_json = {
                    "orig_img_h": int(detection["orig_img_h"]),
                    "orig_img_w": int(detection["orig_img_w"]),
                    "original_image_path": src_path,
                    "pred_boxes": detection.get("pred_boxes") or [],
                    "pred_masks": detection.get("pred_masks") or [],
                }
                for obj_num in objects:
                    context_pil, zoom_pil = visualize(
                        input_json,
                        zoom_in_index=int(obj_num) - 1,
                        mask_alpha=0.15,
                        zoom_mask_alpha=0,
                    )
                    ctx_path = self._save_pil_temp(
                        context_pil, tmp_paths, prefix="examine_ctx_"
                    )
                    zoom_path = self._save_pil_temp(
                        zoom_pil, tmp_paths, prefix="examine_zoom_"
                    )
                    tasks.append(
                        (
                            "object",
                            int(obj_num),
                            _OBJ_SYS_PROMPT,
                            [ctx_path, zoom_path],
                        )
                    )

            results = {}
            workers = max(1, min(self._max_workers, len(tasks)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_key = {
                    pool.submit(
                        self._query_one, sys_prompt, question, image_paths
                    ): (kind, key)
                    for (kind, key, sys_prompt, image_paths) in tasks
                }
                for fut in as_completed(future_to_key):
                    kind, key = future_to_key[fut]
                    try:
                        results[(kind, key)] = fut.result()
                    except Exception:
                        results[(kind, key)] = "NO"
        finally:
            for path in tmp_paths:
                try:
                    os.unlink(path)
                except Exception:
                    pass

        yes_objects = [
            int(n) for n in objects
            if results.get(("object", int(n))) == "YES"
        ]
        no_objects = [
            int(n) for n in objects
            if results.get(("object", int(n))) == "NO"
        ]
        yes_boxes = [
            [float(v) for v in b] for b in boxes_list
            if results.get(("box", tuple(b))) == "YES"
        ]
        no_boxes = [
            [float(v) for v in b] for b in boxes_list
            if results.get(("box", tuple(b))) == "NO"
        ]

        return {
            "yes_count": len(yes_objects) + len(yes_boxes),
            "no_count": len(no_objects) + len(no_boxes),
            "yes_objects": yes_objects,
            "no_objects": no_objects,
            "yes_boxes": yes_boxes,
            "no_boxes": no_boxes,
        }
