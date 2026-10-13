"""Locate objects described by a text query and expose boxes, scores and optional masks.

Visual_Grounding_Tool supports SAM3 segmentation or Grounding DINO detection.
Results include pixel and 0–1000 boxes, Python bindings and model-visible evidence.
"""

import json
import math
from typing import TypedDict

from ._base import BaseCapability, store_selected_binding

PROCESSOR_MODE = "custom"


def _normalized_boxes(boxes, size):
    """Convert pixel xyxy boxes to rounded integers bounded to the 0–1000 scale."""
    if boxes is None or not size or not all(size):
        return None
    return [
        [
            min(max(round(value / size[1 if index % 2 == 0 else 0] * 1000), 0), 1000)
            for index, value in enumerate(box)
        ]
        for box in boxes
    ]


def _safe_int(value, default=0):
    """Coerce a value to an integer, returning default when conversion fails."""
    try:
        return int(value)
    except Exception:
        return default


def process_tool_output(payload, prepared, tool_metadata=None):
    """Render normalized grounding boxes and collect pixel-space detection sidecars.

    Args:
        payload: Execute envelope with the backend result.
        prepared: Evidence fields, including registered image parts and references.
        tool_metadata: Optional metadata accepted by the processor interface.

    Returns:
        A mapping with user_message and processor_info when custom evidence
        is available. None or a fallback record delegates to shared rendering.
    """
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return {
            "processor_info": {
                "mode": PROCESSOR_MODE,
                "status": "fallback",
                "fallback_reason": "non_success_payload",
            }
        }

    result = payload.get("result")
    if not isinstance(result, dict):
        return {
            "processor_info": {
                "mode": PROCESSOR_MODE,
                "status": "fallback",
                "fallback_reason": "result_not_dict",
            }
        }

    image_refs = prepared.get("image_refs") if isinstance(prepared, dict) else []
    image_ref = image_refs[0] if isinstance(image_refs, list) and image_refs else None
    image_parts = prepared.get("image_parts") if isinstance(prepared, dict) else []
    if not isinstance(image_parts, list):
        image_parts = []

    pred_scores = result.get("pred_scores")
    pred_boxes = result.get("pred_boxes", [])
    if isinstance(pred_boxes, list):
        mask_count = len(pred_boxes)
    else:
        mask_count = _safe_int(result.get("num_masks"), default=0)

    tool_name = str(payload.get("tool_name") or "Visual_Grounding_Tool").strip()
    prompt = str(result.get("prompt") or "").strip()

    tool_detection = None
    if mask_count > 0:
        orig_w = result.get("orig_img_w", 1)
        orig_h = result.get("orig_img_h", 1)

        # Attach the per-call (boxes, scores) candidate set to the
        # orchestrator's eval sidecar. The orchestrator sees the
        # `tool_detection` key in this return and forwards it to
        # record_tool_detection — keeps process_fn free of memory access.
        if (
            isinstance(pred_scores, list)
            and pred_scores
            and isinstance(pred_boxes, list)
        ):
            tool_detection = {
                "tool_name": tool_name,
                "category_query": prompt,
                "boxes_xyxy_pixels": [
                    list(b)
                    for b in pred_boxes
                    if isinstance(b, (list, tuple)) and len(b) >= 4
                ],
                "scores": list(pred_scores),
                "image_width": _safe_int(orig_w, default=0),
                "image_height": _safe_int(orig_h, default=0),
            }

        # Agent-visible coords are rounded to integer (no decimals). The
        # sidecar in `tool_detection` still carries full-precision pixel
        # boxes, which the evaluator uses for AP after IoU-matching the
        # agent's echoed (rounded) box back to its source candidate.
        detections = [
            {
                "bbox_2d": [
                    min(max(round(box[0] / orig_w * 1000), 0), 1000),
                    min(max(round(box[1] / orig_h * 1000), 0), 1000),
                    min(max(round(box[2] / orig_w * 1000), 0), 1000),
                    min(max(round(box[3] / orig_h * 1000), 0), 1000),
                ],
                "label": prompt,
            }
            for box in pred_boxes
            if len(box) == 4
        ]
        inner = ",\n".join("\t" + json.dumps(d) for d in detections)
        boxes_block = f"```json\n[\n{inner}\n]```"
        text = (
            f"{tool_name} grounded {mask_count} objects for the prompt {prompt!r}. "
            f"Their corresponding bounding boxes (on a 0-1000 scale, [x1, y1, x2, y2]) are: {boxes_block}."
        )
        if image_ref:
            text += (
                f" The object numbers correspond to the labeled masks rendered in the following image, "
                f"which you can refer to as {image_ref}."
            )
    else:
        text = (
            f"{tool_name} grounded {mask_count} objects for the prompt {prompt!r}."
            + f"Please examine the image carefully to determine whether it contains {prompt!r}. If the image does contain {prompt!r} but {tool_name} did not successfully ground it, you may try calling {tool_name} again with an alternative / simpler prompt or locate it directly by yourself."
        )

    if image_parts:
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": text}] + list(image_parts),
        }
    else:
        user_message = {"role": "user", "content": text}

    out = {
        "user_message": user_message,
        "processor_info": {"mode": PROCESSOR_MODE, "status": "ok"},
    }
    if tool_detection is not None:
        out["tool_detection"] = tool_detection
    return out


PUBLIC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "Visual_Grounding_Tool",
        "description": "Locate and highlight objects in an image that match a text "
        "query. Returns an annotated image with the matched regions highlighted and "
        "numbered, along with bounding boxes and confidence scores.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A short and simple object name or noun phrase, "
                    "such as 'bottle', 'green ball', or 'wooden chair'. Avoid "
                    "relational, comparative, or context-dependent descriptions such "
                    "as 'the tallest man', 'something to eat', or 'the book on the "
                    "left'.",
                },
                "image_id": {
                    "type": "string",
                    "description": "A runtime image ID from Environment Update "
                    "messages, e.g. input_image_1 or tool_generated_image_1.",
                },
            },
            "required": ["query"],
        },
    },
}

FIELD_DESCRIPTIONS = {
    "pred_boxes": "predicted bounding boxes",
    "pred_scores": "prediction confidence scores",
    "pred_masks": "predicted segmentation masks",
    "num_masks": "number of predicted masks",
    "orig_img_h": "original image height",
    "orig_img_w": "original image width",
}


class Output(TypedDict):
    """Canonical grounding output with pixel and normalized boxes.

    size is [height, width]; boxes are pixel xyxy and boxes_normalized use the
    bounded integer 0–1000 xyxy scale. scores and labels align with boxes. SAM3
    masks are COCO RLE counts strings interpreted using size. masks and num_masks
    are None for Grounding DINO. rendered_image is a memory image ID.
    """

    query: str
    model_id: str | None
    size: list[int] | None  # [height, width]
    boxes: list[list[float]] | None  # pixel xyxy
    boxes_normalized: list[list[int]] | None  # 0..1000 xyxy
    scores: list[float] | None
    labels: list[str] | None
    masks: list[str] | None
    num_masks: int | None
    rendered_image: str | None  # Memory image handle


class Capability(BaseCapability):
    """Expose SAM3 or Grounding DINO detections through one public grounding tool."""

    name = "Visual_Grounding_Tool"
    backends = {
        "sam3_segment": "Visual_Grounding_Tool",
        "grounding_dino": "GroundingDINO_Tool",
    }
    schema = PUBLIC_SCHEMA
    process_output = staticmethod(process_tool_output)

    def __init__(self, **kwargs):
        """Select the backend and its SAM3 or Grounding DINO evidence renderer."""
        super().__init__(**kwargs)
        if self.backend == "grounding_dino":
            self.process_output = self._dino_evidence

    def validate_result(self, payload):
        """Require finite, aligned boxes/scores and positive dimensions for detections."""
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ValueError("Grounding result must be an object")
        dino = self.backend == "grounding_dino"
        boxes = result.get("boxes" if dino else "pred_boxes")
        scores = result.get("scores" if dino else "pred_scores")
        size = (
            result.get("size")
            if dino
            else [result.get("orig_img_h"), result.get("orig_img_w")]
        )

        def numeric(value):
            return type(value) in (int, float) and math.isfinite(value)

        if boxes is not None and (
            not isinstance(boxes, list)
            or any(
                not isinstance(box, (list, tuple))
                or len(box) != 4
                or not all(map(numeric, box))
                for box in boxes
            )
        ):
            raise ValueError(
                "Grounding boxes must contain four finite xyxy coordinates"
            )
        if scores is not None and (
            not isinstance(scores, list)
            or not all(map(numeric, scores))
            or boxes is not None
            and len(scores) != len(boxes)
        ):
            raise ValueError("Grounding scores must be finite and aligned with boxes")
        if size is not None and (
            not isinstance(size, (list, tuple))
            or len(size) != 2
            or any(
                value is not None and (not numeric(value) or value <= 0)
                for value in size
            )
        ):
            raise ValueError(
                "Grounding size must be [height, width] with positive dimensions"
            )
        if boxes and (size is None or any(value is None for value in size)):
            raise ValueError("Nonempty grounding boxes require image dimensions")

    @staticmethod
    def _dino_evidence(payload, prepared, tool_metadata=None):
        """Render DINO boxes with explicit mask unavailability and optional score sidecars."""
        result = payload.get("result") or {}
        boxes = _normalized_boxes(result.get("boxes"), result.get("size"))
        text = (
            "Visual_Grounding_Tool returned bounding boxes on the 0–1000 scale "
            "([x1, y1, x2, y2]):\n"
            + json.dumps(boxes, ensure_ascii=False)
            + "\nSegmentation masks are unavailable for this backend."
        )
        size = result.get("size") or [0, 0]
        output = {
            "user_message": {"role": "user", "content": text},
            "processor_info": {"mode": "custom", "status": "ok"},
        }
        if result.get("boxes") and result.get("scores"):
            output["tool_detection"] = {
                "tool_name": "Visual_Grounding_Tool",
                "category_query": result.get("query", ""),
                "boxes_xyxy_pixels": result["boxes"],
                "scores": result["scores"],
                "image_height": size[0],
                "image_width": size[1],
            }
        return output

    def execute(self, call, args, memory):
        """Execute grounding and supply the query when the DINO result omits it."""
        payload = super().execute(call, args, memory)
        if (
            self.backend == "grounding_dino"
            and payload.get("ok")
            and isinstance(payload.get("result"), dict)
        ):
            payload["result"].setdefault("query", call.arguments["query"])
        return payload

    def store_result(self, call, result, memory):
        """Store boxes, scores, mask data and source dimensions as a segment binding."""
        if self.backend == "grounding_dino" and isinstance(result, dict):
            size = result.get("size") or [None, None]
            result = {
                "pred_boxes": result.get("boxes"),
                "pred_scores": result.get("scores"),
                "pred_masks": None,
                "num_masks": None,
                "orig_img_h": size[0],
                "orig_img_w": size[1],
            }
        return store_selected_binding(
            memory,
            call,
            result,
            prefix="segment",
            fields=[
                "pred_boxes",
                "pred_scores",
                "pred_masks",
                "num_masks",
                "orig_img_h",
                "orig_img_w",
            ],
            descriptions=FIELD_DESCRIPTIONS,
        )

    def normalize_result(self, call, result, prepared, binding_refs) -> Output:
        """Return canonical detection fields with both pixel and 0–1000 box coordinates."""
        result = result if isinstance(result, dict) else {}
        dino = self.backend == "grounding_dino"
        size = (
            result.get("size")
            if dino
            else [result.get("orig_img_h"), result.get("orig_img_w")]
        )
        boxes = result.get("boxes" if dino else "pred_boxes")
        scores = result.get("scores" if dino else "pred_scores")
        normalized = _normalized_boxes(boxes, size)
        return {
            "query": result.get("prompt", call.arguments.get("query")),
            "model_id": result.get("model_id", result.get("model")),
            "size": size,
            "boxes": boxes,
            "boxes_normalized": normalized,
            "scores": scores,
            "labels": result.get("labels")
            if dino
            else [call.arguments["query"] for _ in boxes or []],
            "masks": None if dino else result.get("pred_masks"),
            "num_masks": None if dino else result.get("num_masks"),
            "rendered_image": next(iter(prepared["image_refs"]), None),
        }
