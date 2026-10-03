"""Camera_Trajectory_Tool output post-processing for agent-visible messages."""

PROCESSOR_MODE = "custom"


def _build_bev_caption(reference_image_id, image_ref):
    frame = reference_image_id or "the reference image"
    return (
        f"Bird's-eye view of the camera trajectory on {frame}'s X-Z plane is "
        f"rendered in the following image - refer as {image_ref} "
        "(dots = camera positions, forward-pointing arrows = facing direction)."
    )


def _insert_after_preamble(text, caption):
    """Insert `caption` between the coordinate-system preamble (the first
    paragraph) and the rest of the text. Falls back to prepending if the text
    has no blank-line separator."""
    if "\n\n" in text:
        head, tail = text.split("\n\n", 1)
        return f"{head}\n\n{caption}\n\n{tail}"
    return f"{caption}\n\n{text}"


def process_tool_output(payload, prepared, tool_metadata=None):
    del tool_metadata
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict) or not payload.get("ok"):
        return {"processor_info": {"mode": PROCESSOR_MODE, "status": "fallback"}}

    tool_name = str(payload.get("tool_name") or "Camera_Trajectory_Tool")
    text = result.get("text_summary")
    if not isinstance(text, str) or not text.strip():
        text = (
            f"{tool_name} produced no interpretable trajectory "
            "(text_summary missing)."
        )

    image_parts = prepared.get("image_parts") if isinstance(prepared, dict) else []
    if not isinstance(image_parts, list):
        image_parts = []
    image_refs = prepared.get("image_refs") if isinstance(prepared, dict) else []
    image_ref = (
        image_refs[0]
        if isinstance(image_refs, list) and image_refs
        else None
    )
    if image_ref:
        caption = _build_bev_caption(result.get("reference_image_id"), image_ref)
        text = _insert_after_preamble(text, caption)

    if image_parts:
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": text}] + list(image_parts),
        }
    else:
        user_message = {"role": "user", "content": text}

    return {
        "user_message": user_message,
        "processor_info": {"mode": PROCESSOR_MODE, "status": "ok"},
    }


from ._base import BaseCapability

PUBLIC_SCHEMA = {'type': 'function',
 'function': {'name': 'Camera_Trajectory_Tool',
              'description': 'Estimate per-image camera motion and viewpoint from an ordered '
                             'sequence of images of the same scene (>= 2 images required; the '
                             "first image is the reference frame). Returns a bird's-eye-view image "
                             "of camera positions on the reference frame's X-Z plane plus a "
                             'per-image text summary (movement and rotation) and structured '
                             'numeric evidence (camera_center_delta, forward_direction, yaw_deg, '
                             "pitch_deg) — all expressed in the reference image's camera frame (+X "
                             '= right, +Y = down, +Z = forward). Use for questions about how the '
                             'camera/viewpoint moved or rotated across frames. For single-image '
                             'depth, use Depth_Estimation_Tool. Caveats: scale is metric only up '
                             "to model's calibration; the reference image must share enough scene "
                             'overlap with the other views; raw 3x3 intrinsics and 3x4 '
                             'world-to-camera extrinsics are also returned for downstream Python '
                             'use.',
              'parameters': {'type': 'object',
                             'properties': {'image_ids': {'type': 'array',
                                                          'items': {'type': 'string'},
                                                          'description': 'Ordered runtime image '
                                                                         'IDs from Environment '
                                                                         'Update messages, e.g. '
                                                                         '["input_image_1", '
                                                                         '"input_image_2"]. The '
                                                                         'framework resolves them '
                                                                         'to image bytes before '
                                                                         'dispatch.'}},
                             'additionalProperties': False,
                             'required': ['image_ids']}}}

from ._base import store_selected_binding
FIELD_DESCRIPTIONS = {'intrinsics': 'per-image 3x3 camera intrinsic matrices (ndarray '
               'shape=(N,3,3))',
 'extrinsics': 'per-image 3x4 world-to-camera matrices (OpenCV/COLMAP), '
               'ndarray shape=(N,3,4)',
 'image_ids': 'ordered runtime image IDs aligned with intrinsics/extrinsics',
 'num_images': 'number of images processed',
 'cameras': 'per-image dicts (in reference frame): camera_center_delta, '
            'movement_summary, forward_direction, rotation_summary, yaw_deg, '
            'pitch_deg',
 'coordinate_system': 'human-readable coordinate convention used for cameras',
 'reference_image_id': 'image_id of the reference frame (always image_ids[0])'}

from typing import TypedDict


class ArrayRef(TypedDict):
    binding: str
    field: str


class Output(TypedDict):
    binding: str | None
    intrinsics: ArrayRef | None  # float32 N x 3 x 3
    extrinsics: ArrayRef | None  # float32 N x 3 x 4, world-to-camera
    image_ids: list[str] | None
    num_images: int | None
    cameras: list[dict] | None
    coordinate_system: str | None
    reference_image_id: str | None
    text_summary: str | None
    image_sizes: list[list[int]] | None
    model_id: str | None
    visualization: str | None


class Capability(BaseCapability):
    name = 'Camera_Trajectory_Tool'
    backends = {'camera_trajectory': 'Camera_Trajectory_Tool'}
    schema = PUBLIC_SCHEMA
    process_output = staticmethod(process_tool_output)

    def store_result(self, call, result, memory):
        def convert(value):
            import numpy as np
            for key in ("intrinsics", "extrinsics"):
                if key in value:
                    try:
                        value[key] = np.asarray(value[key], dtype=np.float32)
                    except Exception:
                        pass
            return value
        return store_selected_binding(memory, call, result, prefix="camera_trajectory",
            fields=["intrinsics", "extrinsics", "image_ids", "num_images", "cameras",
                    "coordinate_system", "reference_image_id"],
            descriptions=FIELD_DESCRIPTIONS, convert=convert)

    def normalize_result(self, call, result, prepared, binding_refs) -> Output:
        result = result if isinstance(result, dict) else {}
        binding = next(iter(binding_refs), None)
        return {"binding": binding,
            **{key: {"binding": binding, "field": key} if binding and result.get(key) is not None else None
               for key in ("intrinsics", "extrinsics")},
            **{key: result.get(key) for key in ("image_ids", "num_images", "cameras",
                "coordinate_system", "reference_image_id", "text_summary", "image_sizes", "model_id")},
            "visualization": next(iter(prepared["image_refs"]), None)}
