"""Recover camera poses and describe movement across two or more ordered views.

The Camera_Trajectory_Tool client calls DA3's pose endpoint, interprets camera
centers and viewing directions in the first camera's frame, and renders an
optional bird's-eye diagram.
"""
import mimetypes
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from openvlharness.services.tools.base import BaseTool, ToolArgumentError
from openvlharness.services.tools.http_utils import raise_for_status_with_detail
from openvlharness.services.tools.image_utils import decode_base64_image_data

from .interpret import (
    coordinate_system_text,
    interpret_camera_trajectory,
    render_bird_eye_view_image,
    render_camera_motion_message,
)

TOOL_NAME = "Camera_Trajectory_Tool"
DEFAULT_SERVER_URL = "http://127.0.0.1:8087"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Ordered local image paths or runtime image IDs from Environment "
                "Update messages. Provide at least 2 ordered images of the same "
                "scene. The first image is used as the reference frame."
            ),
        },
        "image_base64s": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Resolved base64 image data; populated by the runtime. Provide "
                "at least 2 ordered images. The first image is the reference."
            ),
        },
        "_image_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Original runtime image IDs preserved by the resolver.",
        },
    },
    "oneOf": [
        {"required": ["image_paths"]},
        {"required": ["image_base64s"]},
    ],
    "additionalProperties": False,
}


class Camera_Trajectory_Tool(BaseTool):
    """Ordered multi-image pose estimation and camera-motion interpretation."""
    def __init__(self):
        """Initialize tool metadata and its concrete input/output contract."""
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Estimate per-image camera motion and viewpoint from an ordered "
                "sequence of images of the same scene (>= 2 images required; the "
                "first image is the reference frame). Returns a bird's-eye-view "
                "image of camera positions on the reference frame's X-Z plane "
                "plus a per-image text summary (movement and rotation) and "
                "structured numeric evidence (camera_center_delta, "
                "forward_direction, yaw_deg, pitch_deg) — all expressed in the "
                "reference image's camera frame (+X = right, +Y = down, "
                "+Z = forward). Use for questions about how the camera/viewpoint "
                "moved or rotated across frames. For single-image depth, use "
                "Depth_Estimation_Tool. Caveats: scale is metric only up to "
                "model's calibration; the reference image must share enough scene "
                "overlap with the other views; raw 3x3 intrinsics and 3x4 "
                "world-to-camera extrinsics are also returned for downstream "
                "Python use."
            ),
            tool_version="0.2.0",
            input_types={
                "image_paths": "list[str] - ordered local image paths or runtime image IDs",
                "image_base64s": "list[str] - resolved runtime image bytes",
            },
            input_schema=INPUT_SCHEMA,
            output_type=(
                "dict - tool_name, image_ids, num_images, image_sizes, "
                "coordinate_system, reference_image_id, cameras "
                "(list[dict] per image: camera_center_delta, movement_summary, "
                "forward_direction, rotation_summary, yaw_deg, pitch_deg), "
                "text_summary (model-facing), bird_eye_view_image_base64 "
                "(PNG of X-Z map), intrinsics (list[N, 3, 3]), "
                "extrinsics (list[N, 3, 4]), model_id, device"
            ),
            demo_commands=[
                {
                    "command": (
                        'Camera_Trajectory_Tool('
                        'image_paths=["input_image_1", "input_image_2"])'
                    ),
                    "description": (
                        "Determine how the camera moved and rotated between two "
                        "views of the same scene."
                    ),
                },
                {
                    "command": (
                        'Camera_Trajectory_Tool('
                        'image_paths=["input_image_1", "input_image_2", "input_image_3"])'
                    ),
                    "description": (
                        "Summarize the camera trajectory across a short "
                        "multi-view sequence relative to the first frame."
                    ),
                },
            ],
            user_metadata={
                "source": "DepthAnything3",
                "status": "implemented",
                "server_env": "DEPTH_ESTIMATION_SERVER_URL",
                "default_server_url": DEFAULT_SERVER_URL,
                "notes": (
                    "Calls the DA3 server /v1/pose with multiple images, then "
                    "interprets the world-to-camera extrinsics into a "
                    "coordinate-aware trajectory description."
                ),
            },
        )

    def sanity_check(self):
        """Probe the configured health endpoint and return issue strings, or an empty list."""
        server_url = os.getenv(
            "DEPTH_ESTIMATION_SERVER_URL", DEFAULT_SERVER_URL
        ).rstrip("/")
        health_url = f"{server_url}/health"
        try:
            response = requests.get(health_url, timeout=5)
        except Exception as exc:
            return [
                f"Camera trajectory server unreachable at {health_url}: {exc}"
            ]
        if response.status_code != 200:
            return [
                f"Camera trajectory server unhealthy at {health_url}: "
                f"{response.status_code}"
            ]
        return []

    def _build_files_from_paths(
        self, image_paths: List[str]
    ) -> List[Tuple[str, Tuple[str, bytes, str]]]:
        """Read ordered local images into a multipart upload list, preserving MIME types."""
        files = []
        for path in image_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"image_path not found: {path}")
            content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            with open(path, "rb") as handle:
                files.append(
                    ("images", (os.path.basename(path), handle.read(), content_type))
                )
        return files

    def _build_files_from_base64s(
        self, image_base64s: List[str]
    ) -> List[Tuple[str, Tuple[str, bytes, str]]]:
        """Decode ordered inline images into a multipart upload list."""
        files = []
        for idx, encoded in enumerate(image_base64s):
            image_bytes, content_type = decode_base64_image_data(encoded)
            files.append(
                (
                    "images",
                    (
                        f"image_{idx + 1}",
                        image_bytes,
                        content_type or "application/octet-stream",
                    ),
                )
            )
        return files

    def execute(
        self,
        image_paths: Optional[List[str]] = None,
        image_base64s: Optional[List[str]] = None,
        _image_ids: Optional[List[str]] = None,
        server_url: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> Dict[str, Any]:
        """Estimate ordered camera poses and describe motion relative to the first image.

        Args:
            image_paths: At least two local paths, exclusive with image_base64s.
            image_base64s: At least two inline image payloads in temporal/view order.
            _image_ids: Optional source IDs preserved during capability resolution.
            server_url: DA3 service base, otherwise resolved from the environment.
            timeout_seconds: Per-request HTTP timeout.

        Returns:
            Backend intrinsics/extrinsics plus ordered image IDs, camera motion records,
            coordinate-system text, a text summary and an optional PNG bird's-eye view.

        Raises:
            ValueError: Sources are missing, ambiguous, too short or contain invalid data.
            FileNotFoundError: A supplied image file is missing.
            RuntimeError: The pose service returns an HTTP error.
        """
        if not image_paths and not image_base64s:
            raise ToolArgumentError("image_paths or image_base64s is required")
        if image_paths and image_base64s:
            raise ToolArgumentError(
                "Provide exactly one of image_paths or image_base64s"
            )

        if image_paths is not None:
            if not isinstance(image_paths, list) or len(image_paths) < 2:
                raise ToolArgumentError(
                    "Camera_Trajectory_Tool requires at least 2 image_paths. "
                    "For single-image depth, use Depth_Estimation_Tool."
                )
            files = self._build_files_from_paths(image_paths)
            source_ids = _image_ids or image_paths
        else:
            if not isinstance(image_base64s, list) or len(image_base64s) < 2:
                raise ToolArgumentError(
                    "Camera_Trajectory_Tool requires at least 2 image_base64s. "
                    "For single-image depth, use Depth_Estimation_Tool."
                )
            files = self._build_files_from_base64s(image_base64s)
            source_ids = _image_ids or [
                f"image_{idx + 1}" for idx in range(len(image_base64s))
            ]

        resolved_server_url = (
            server_url
            or os.getenv("DEPTH_ESTIMATION_SERVER_URL", DEFAULT_SERVER_URL)
        ).rstrip("/")
        endpoint = f"{resolved_server_url}/v1/pose"

        response = requests.post(endpoint, files=files, timeout=timeout_seconds)
        raise_for_status_with_detail(
            response, prefix="Camera trajectory server error"
        )
        result = response.json()

        image_ids = [str(image_id) for image_id in source_ids]
        reference_image_id = image_ids[0]
        extrinsics_arr = np.asarray(result["extrinsics"], dtype=np.float32)
        cameras = interpret_camera_trajectory(extrinsics_arr, image_ids)

        bird_eye_view_image_base64 = render_bird_eye_view_image(
            cameras, reference_image_id=reference_image_id
        )
        text_summary = render_camera_motion_message(
            cameras,
            reference_image_id=reference_image_id,
        )

        result["tool_name"] = TOOL_NAME
        result["image_ids"] = image_ids
        result["coordinate_system"] = coordinate_system_text(reference_image_id)
        result["reference_image_id"] = reference_image_id
        result["cameras"] = cameras
        result["text_summary"] = text_summary
        if bird_eye_view_image_base64:
            result["bird_eye_view_image_base64"] = bird_eye_view_image_base64
        return result
