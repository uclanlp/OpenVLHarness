import os
from typing import Any, Dict, List, Optional

import requests

from tools.base import BaseTool
from tools.http_utils import raise_for_status_with_detail
from tools.image_utils import decode_base64_image_data
from tools.spatial_utils import (
    build_tracking_summary,
    choose_best_candidate,
    decode_depth_array,
    extract_candidates,
    movement_direction,
)


TOOL_NAME = "Tracking_Tool"
DEFAULT_SAM3_SERVER_URL = "http://127.0.0.1:8086"
DEFAULT_DEPTH_SERVER_URL = "http://127.0.0.1:8087"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {
            "type": "string",
            "description": "Short object/entity noun phrase to track. Avoid relational modifiers or deictic references.",
        },
        "image_paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Ordered local image paths or runtime image IDs from Environment "
                "Update messages. Provide at least 2 ordered images."
            ),
        },
        "image_base64s": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Resolved base64 image data; populated by the runtime. Provide "
                "at least 2 ordered images."
            ),
        },
        "_image_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Original runtime image IDs preserved by the resolver.",
        },
    },
    "required": ["target"],
    "oneOf": [
        {"required": ["image_paths"]},
        {"required": ["image_base64s"]},
    ],
    "additionalProperties": False,
}


class Tracking_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Track a target object across an ordered sequence of images. "
                "Requires at least 2 input images. For single-image questions "
                "about the position or identity of an object, use "
                "Visual_Grounding_Tool instead. Use when a question involves "
                "how an object or the viewpoint moves across multiple frames — "
                "trajectory direction, approaching or receding, before/after "
                "positions, or which frame an object appears or disappears in. "
                "Returns per-frame detections, 2D movement direction, depth "
                "change, and ambiguity metadata. Image-plane directions only; "
                "if the target is stationary, infer camera motion as the inverse."
            ),
            tool_version="0.1.0",
            input_types={
                "target": "str - short object/entity noun phrase",
                "image_paths": "list[str] - ordered local image paths or runtime image IDs",
                "image_base64s": "list[str] - resolved runtime image bytes",
            },
            input_schema=INPUT_SCHEMA,
            output_type=(
                "dict - status, target, image_ids, detections, tracks, movement, "
                "summary, ambiguities, grounding_images"
            ),
            demo_commands=[
                {
                    "command": (
                        'Tracking_Tool(target="person in red", '
                        'image_ids=["input_image_1", "input_image_2", "input_image_3"])'
                    ),
                    "description": (
                        "Decide whether the person moved left or right across "
                        "three sequential frames."
                    ),
                },
                {
                    "command": (
                        'Tracking_Tool(target="cup", '
                        'image_ids=["input_image_1", "input_image_2"])'
                    ),
                    "description": (
                        "Compare a before/after pair to detect whether the cup "
                        "moved or was displaced."
                    ),
                },
                {
                    "command": (
                        'Tracking_Tool(target="white SUV", '
                        'image_ids=["input_image_1", "input_image_2", "input_image_3"])'
                    ),
                    "description": (
                        "Match the same object across multi-camera views and "
                        "report depth/position change."
                    ),
                },
            ],
            user_metadata={"status": "implemented", "composes": [
                "Visual_Grounding_Tool", "Depth_Estimation_Tool"
            ]},
        )
    def _post_image_request(
        self,
        *,
        server_url: str,
        endpoint_path: str,
        image_base64: str,
        data: Dict[str, Any],
        timeout_seconds: int,
        error_prefix: str,
    ) -> Dict[str, Any]:
        image_bytes, content_type = decode_base64_image_data(image_base64)
        files = {
            "image": ("image", image_bytes, content_type or "application/octet-stream"),
        }
        response = requests.post(
            f"{server_url.rstrip('/')}{endpoint_path}",
            files=files,
            data=data,
            timeout=timeout_seconds,
        )
        raise_for_status_with_detail(response, prefix=error_prefix)
        return response.json()

    def _run_grounding(
        self,
        *,
        target: str,
        image_base64: str,
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        server_url = os.getenv("SAM3_SERVER_URL", DEFAULT_SAM3_SERVER_URL).rstrip("/")
        return self._post_image_request(
            server_url=server_url,
            endpoint_path="/v1/segment",
            image_base64=image_base64,
            data={"prompt": target},
            timeout_seconds=timeout_seconds,
            error_prefix="SAM3 server error",
        )

    def _run_depth(
        self,
        *,
        image_base64: str,
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        server_url = os.getenv(
            "DEPTH_ESTIMATION_SERVER_URL", DEFAULT_DEPTH_SERVER_URL
        ).rstrip("/")
        return self._post_image_request(
            server_url=server_url,
            endpoint_path="/v1/depth",
            image_base64=image_base64,
            data={},
            timeout_seconds=timeout_seconds,
            error_prefix="Depth estimation server error",
        )

    def _run_grounding_and_depth(
        self,
        target: str,
        image_base64: str,
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        grounding = self._run_grounding(
            target=target,
            image_base64=image_base64,
            timeout_seconds=timeout_seconds,
        )
        depth_result = self._run_depth(
            image_base64=image_base64,
            timeout_seconds=timeout_seconds,
        )
        depth = decode_depth_array(depth_result)
        return {
            "grounding": grounding,
            "depth": depth,
            "candidates": extract_candidates(grounding, depth),
        }

    @staticmethod
    def _match_candidate(
        previous: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        prev_center = previous.get("normalized_center") or [0.5, 0.5]
        prev_box = previous.get("box") or [0, 0, 1, 1]
        prev_area = max(
            (float(prev_box[2]) - float(prev_box[0]))
            * (float(prev_box[3]) - float(prev_box[1])),
            1.0,
        )
        prev_depth = previous.get("depth_value")

        def score(candidate: Dict[str, Any]) -> float:
            center = candidate.get("normalized_center") or [0.5, 0.5]
            dx = float(center[0]) - float(prev_center[0])
            dy = float(center[1]) - float(prev_center[1])
            center_score = max(0.0, 1.0 - ((dx * dx + dy * dy) ** 0.5))

            area = max(float(candidate.get("box_area") or 1.0), 1.0)
            size_ratio = min(area, prev_area) / max(area, prev_area)

            depth_score = 0.5
            cand_depth = candidate.get("depth_value")
            if isinstance(prev_depth, (int, float)) and isinstance(cand_depth, (int, float)):
                denom = max(abs(float(prev_depth)), abs(float(cand_depth)), 1.0)
                depth_score = max(0.0, 1.0 - abs(float(prev_depth) - float(cand_depth)) / denom)

            confidence = float(candidate.get("confidence") or 0.0)
            return (
                0.45 * confidence
                + 0.25 * size_ratio
                + 0.20 * center_score
                + 0.10 * depth_score
            )

        return max(candidates, key=score)

    def execute(
        self,
        target=None,
        image_base64s=None,
        image_paths=None,
        _image_ids=None,
        timeout_seconds=None,
    ):
        timeout_seconds = int(timeout_seconds or 300)
        if not isinstance(target, str) or not target.strip():
            return {
                "tool_name": TOOL_NAME,
                "status": "invalid_input",
                "summary": "`target` must be a non-empty string.",
            }
        target = target.strip()

        if image_base64s is None:
            image_base64s = []
        if not isinstance(image_base64s, list) or len(image_base64s) < 2:
            return {
                "tool_name": TOOL_NAME,
                "target": target,
                "status": "invalid_input",
                "summary": (
                    "`image_ids` must resolve to at least two ordered images. "
                    "If you only have one image, switch to Visual_Grounding_Tool "
                    "to locate the object instead."
                ),
            }

        source_image_ids = _image_ids or image_paths
        if not isinstance(source_image_ids, list) or len(source_image_ids) != len(image_base64s):
            source_image_ids = [f"image_{idx + 1}" for idx in range(len(image_base64s))]

        detections: List[Dict[str, Any]] = []
        grounding_images: List[Dict[str, Any]] = []
        ambiguities: List[Dict[str, Any]] = []

        try:
            for image_id, image_base64 in zip(source_image_ids, image_base64s):
                bundle = self._run_grounding_and_depth(
                    target,
                    image_base64,
                    timeout_seconds,
                )
                grounding = bundle["grounding"]
                candidates = bundle["candidates"]
                detections.append({"image_id": image_id, "candidates": candidates})
                rendered = grounding.get("rendered_image_b64") if isinstance(grounding, dict) else None
                if rendered:
                    grounding_images.append(
                        {"image_id": image_id, "rendered_image_base64": rendered}
                    )
                if len(candidates) > 1:
                    ambiguities.append(
                        {
                            "image_id": image_id,
                            "candidate_count": len(candidates),
                            "message": "Multiple target candidates were found.",
                        }
                    )
        except Exception as exc:
            return {
                "tool_name": TOOL_NAME,
                "target": target,
                "image_ids": source_image_ids,
                "status": "tool_error",
                "summary": f"Tracking failed while running grounding/depth: {exc}",
            }

        first_candidates = detections[0]["candidates"] if detections else []
        first = choose_best_candidate(first_candidates)
        if first is None:
            return {
                "tool_name": TOOL_NAME,
                "target": target,
                "image_ids": source_image_ids,
                "status": "not_found",
                "detections": detections,
                "grounding_images": grounding_images,
                "summary": f"No reliable {target} detection was found in {source_image_ids[0]}.",
                "ambiguities": ambiguities,
            }

        track = [dict(first, image_id=source_image_ids[0])]
        previous = first
        missing_ids: List[str] = []
        for idx, detection in enumerate(detections[1:], start=1):
            matched = self._match_candidate(previous, detection["candidates"])
            if matched is None:
                missing_ids.append(source_image_ids[idx])
                continue
            track_item = dict(matched, image_id=source_image_ids[idx])
            track.append(track_item)
            previous = matched

        if len(track) < 2:
            return {
                "tool_name": TOOL_NAME,
                "target": target,
                "image_ids": source_image_ids,
                "status": "not_found",
                "detections": detections,
                "grounding_images": grounding_images,
                "tracks": track,
                "summary": f"The {target} was not reliably matched beyond the first image.",
                "ambiguities": ambiguities + [
                    {"image_ids": missing_ids, "message": "No candidate found."}
                ],
            }

        movement = movement_direction(track[0], track[-1])
        summary = build_tracking_summary(target, movement)
        status = "partial_track" if missing_ids else "tracked"
        if missing_ids:
            ambiguities.append(
                {
                    "image_ids": missing_ids,
                    "message": "The target was not detected in every image.",
                }
            )

        return {
            "tool_name": TOOL_NAME,
            "target": target,
            "image_ids": source_image_ids,
            "status": status,
            "detections": detections,
            "tracks": track,
            "movement": movement,
            "summary": summary,
            "ambiguities": ambiguities,
            "grounding_images": grounding_images,
        }
