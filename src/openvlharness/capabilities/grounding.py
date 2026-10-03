"""SAM3-specific output post-processing for agent-visible messages."""

import json

PROCESSOR_MODE = "custom"


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def process_tool_output(payload, prepared, tool_metadata=None):
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return {
            "processor_info": {
                "mode": PROCESSOR_MODE, "status": "fallback",
                "fallback_reason": "non_success_payload",
            }
        }

    result = payload.get("result")
    if not isinstance(result, dict):
        return {
            "processor_info": {
                "mode": PROCESSOR_MODE, "status": "fallback",
                "fallback_reason": "result_not_dict",
            }
        }

    image_refs = prepared.get("image_refs") if isinstance(prepared, dict) else []
    image_ref = image_refs[0] if isinstance(image_refs, list) and image_refs else None
    image_parts = prepared.get("image_parts") if isinstance(prepared, dict) else []
    if not isinstance(image_parts, list):
        image_parts = []

    pred_scores = result.get("pred_scores")
    if isinstance(pred_scores, list):
        mask_count = len(pred_scores)
    else:
        mask_count = _safe_int(result.get("num_masks"), default=0)

    tool_name = str(payload.get("tool_name") or "Visual_Grounding_Tool").strip()
    prompt = str(result.get("prompt") or "").strip()

    tool_detection = None
    if mask_count > 0 and image_ref:
        pred_boxes = result.get("pred_boxes", [])
        orig_w = result.get("orig_img_w", 1)
        orig_h = result.get("orig_img_h", 1)

        # Phase 2: stash per-call (boxes, scores) candidate set on the
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
                    list(b) for b in pred_boxes
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
            f"Their corresponding bounding boxes (on a 0-1000 scale, [x1, y1, x2, y2]) are: {boxes_block}. "
            f"The object numbers correspond to the labeled masks rendered in the following image, which you can refer to as {image_ref}."
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


from ._base import BaseCapability

PUBLIC_SCHEMA = {'type': 'function',
 'function': {'name': 'Visual_Grounding_Tool',
              'description': 'Locate and highlight objects in an image that match a text query. '
                             'Returns an annotated image with the matched regions highlighted and '
                             'numbered, along with bounding boxes and confidence scores.',
              'parameters': {'type': 'object',
                             'properties': {'query': {'type': 'string',
                                                      'description': 'A short and simple object '
                                                                     'name or noun phrase, such as '
                                                                     "'bottle', 'green ball', or "
                                                                     "'wooden chair'. Avoid "
                                                                     'relational, comparative, or '
                                                                     'context-dependent '
                                                                     "descriptions such as 'the "
                                                                     "tallest man', 'something to "
                                                                     "eat', or 'the book on the "
                                                                     "left'."},
                                            'image_id': {'type': 'string',
                                                         'description': 'A runtime image ID from '
                                                                        'Environment Update '
                                                                        'messages, e.g. '
                                                                        'input_image_1 or '
                                                                        'tool_generated_image_1.'}},
                             'required': ['query']}}}

from ._base import store_selected_binding
FIELD_DESCRIPTIONS = {'pred_boxes': 'predicted bounding boxes',
 'pred_scores': 'prediction confidence scores',
 'pred_masks': 'predicted segmentation masks',
 'num_masks': 'number of predicted masks',
 'orig_img_h': 'original image height',
 'orig_img_w': 'original image width'}

from typing import TypedDict


class Output(TypedDict):
    query: str
    model_id: str | None
    size: list[int] | None  # [height, width]
    boxes: list[list[float]] | None  # pixel xyxy
    boxes_normalized: list[list[int]] | None  # 0..1000 xyxy
    scores: list[float] | None
    labels: list[str] | None
    masks: list[dict] | None
    num_masks: int | None
    rendered_image: str | None  # Memory image handle


class Capability(BaseCapability):
    name = 'Visual_Grounding_Tool'
    backends = {'sam3_segment': 'Visual_Grounding_Tool', 'grounding_dino': 'GroundingDINO_Tool'}
    schema = PUBLIC_SCHEMA
    process_output = staticmethod(process_tool_output)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.backend == "grounding_dino":
            self.process_output = self._dino_evidence

    @staticmethod
    def _dino_evidence(payload, prepared, tool_metadata=None):
        result = payload.get("result") or {}
        boxes = result.get("boxes_normalized")
        text = ("Visual_Grounding_Tool returned bounding boxes on the 0–1000 scale "
                "([x1, y1, x2, y2]):\n" + json.dumps(boxes, ensure_ascii=False)
                + "\nSegmentation masks are unavailable for this backend.")
        size = result.get("size") or [0, 0]
        output = {"user_message": {"role": "user", "content": text},
                  "processor_info": {"mode": "custom", "status": "ok"}}
        if result.get("boxes") and result.get("scores"):
            output["tool_detection"] = {"tool_name": "Visual_Grounding_Tool",
                "category_query": result.get("query", ""), "boxes_xyxy_pixels": result["boxes"],
                "scores": result["scores"], "image_height": size[0], "image_width": size[1]}
        return output

    def execute(self, call, args, memory):
        payload = super().execute(call, args, memory)
        if self.backend == "grounding_dino" and payload.get("ok") and isinstance(payload.get("result"), dict):
            payload["result"].setdefault("query", call.arguments["query"])
        return payload

    def store_result(self, call, result, memory):
        if self.backend == "grounding_dino" and isinstance(result, dict):
            size = result.get("size") or [None, None]
            result = {"pred_boxes": result.get("boxes"), "pred_scores": result.get("scores"),
                "pred_masks": None, "num_masks": None, "orig_img_h": size[0], "orig_img_w": size[1]}
        return store_selected_binding(memory, call, result, prefix="segment",
            fields=["pred_boxes", "pred_scores", "pred_masks", "num_masks", "orig_img_h", "orig_img_w"],
            descriptions=FIELD_DESCRIPTIONS)

    def normalize_result(self, call, result, prepared, binding_refs) -> Output:
        result = result if isinstance(result, dict) else {}
        dino = self.backend == "grounding_dino"
        size = result.get("size") if dino else [result.get("orig_img_h"), result.get("orig_img_w")]
        boxes = result.get("boxes" if dino else "pred_boxes")
        scores = result.get("scores" if dino else "pred_scores")
        normalized = result.get("boxes_normalized") if dino else None
        if not dino and size and all(size) and boxes is not None:
            normalized = [[min(max(round(v / size[1 if i % 2 == 0 else 0] * 1000), 0), 1000)
                for i, v in enumerate(box)] for box in boxes]
        return {"query": result.get("prompt", call.arguments.get("query")),
            "model_id": result.get("model_id", result.get("model")), "size": size,
            "boxes": boxes, "boxes_normalized": normalized, "scores": scores,
            "labels": result.get("labels") if dino else [call.arguments["query"] for _ in boxes or []],
            "masks": None if dino else result.get("pred_masks"),
            "num_masks": None if dino else result.get("num_masks"),
            "rendered_image": next(iter(prepared["image_refs"]), None)}
