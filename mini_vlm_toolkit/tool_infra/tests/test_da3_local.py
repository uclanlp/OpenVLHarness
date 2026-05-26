"""Quick test: exercise /v1/pose and Camera_Trajectory_Tool via the router.

Prereqs:
  - DA3 server (defaults to :8087)
        bash mini_vlm_toolkit/tool_infra/src/tools/depth_estimation/run_depth_estimation_server.sh
  - Tool router (defaults to :8101) hosting Camera_Trajectory_Tool.
"""

import base64
import os
from pathlib import Path

import numpy as np
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
# IMAGE_PATHS = [DATA_DIR / "pose_1_0.png", DATA_DIR / "pose_1_1.png", DATA_DIR / "pose_1_2.png", DATA_DIR / "pose_1_3.png"]
IMAGE_PATHS = [DATA_DIR / "pose_2_0.png", DATA_DIR / "pose_2_1.png"]
SERVER_URL = os.getenv("DEPTH_ESTIMATION_SERVER_URL", "http://127.0.0.1:8087").rstrip("/")
TOOL_ROUTER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101").rstrip("/")
TOOL_NAME = "Camera_Trajectory_Tool"


def _make_runtime_args(image_paths):
    """Mimic the agent-side image resolver: emit image_base64s as data URLs
    and the corresponding pseudo runtime IDs in _image_ids."""
    image_base64s = []
    image_ids = []
    for idx, path in enumerate(image_paths, start=1):
        with open(path, "rb") as f:
            raw = base64.b64encode(f.read()).decode("ascii")
        mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
        image_base64s.append(f"data:{mime};base64,{raw}")
        image_ids.append(f"input_image_{idx}")
    return (
        {"image_base64s": image_base64s, "_image_ids": image_ids},
        image_ids,
    )


def camera_center_to_world(extrinsic):
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    R = extrinsic[:, :3]
    t = extrinsic[:, 3]
    return np.round(-R.T @ t, 3)


def test_server_pose_endpoint():
    files = []
    handles = []
    try:
        for path in IMAGE_PATHS:
            handle = open(path, "rb")
            handles.append(handle)
            files.append(("images", (path.name, handle, "image/jpeg")))
        resp = requests.post(f"{SERVER_URL}/v1/pose", files=files, timeout=120)
    finally:
        for handle in handles:
            handle.close()

    resp.raise_for_status()
    result = resp.json()
    intrinsics = np.asarray(result["intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(result["extrinsics"], dtype=np.float32)
    print("Server /v1/pose response keys:", list(result.keys()))
    print(f"intrinsics shape={intrinsics.shape}, extrinsics shape={extrinsics.shape}")
    print(f"num_images={result['num_images']}, image_sizes={result['image_sizes']}")
    for idx, extrinsic in enumerate(extrinsics):
        center = camera_center_to_world(extrinsic)
        print(f"  image {idx} camera center (world): {center}")
        print(f"  image {idx} right direction (camera): {np.round(extrinsic[0, :3], 3)}")
        print(f"  image {idx} down direction (camera): {np.round(extrinsic[1, :3], 3)}")
        print(f"  image {idx} forward direction (world): {np.round(extrinsic[2, :3], 3)}")
    return result


def test_tool_via_router():
    args, expected_ids = _make_runtime_args(IMAGE_PATHS)
    response = requests.post(
        f"{TOOL_ROUTER_URL}/execute",
        json={"tool_name": TOOL_NAME, "args": args},
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    assert payload.get("ok") is True, f"router returned not-ok payload: {payload}"

    result = payload["result"]
    assert result.get("image_ids") == expected_ids, (
        f"expected image_ids={expected_ids} from runtime _image_ids; "
        f"got {result.get('image_ids')!r}"
    )

    text_summary = result.get("text_summary")
    assert isinstance(text_summary, str) and text_summary.strip(), (
        f"expected non-empty text_summary in result; got {text_summary!r}"
    )
    assert expected_ids[0] in text_summary, (
        f"expected reference id {expected_ids[0]!r} in text_summary; "
        f"got {text_summary!r}"
    )

    # If the router didn't scrub the inline image, save it for visual inspection.
    bev_b64 = result.get("bird_eye_view_image_base64")
    if isinstance(bev_b64, str) and bev_b64:
        raw = base64.b64decode(bev_b64)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", "bird_eye_view payload is not PNG"
        out_path = "data/camera_trajectory_bev.png"
        with open(out_path, "wb") as f:
            f.write(raw)
        print(f"Saved bird-eye-view PNG -> {out_path}")
    else:
        image_refs = payload.get("image_refs") or []
        if image_refs:
            print(f"Bird-eye-view image registered as: {image_refs}")
        else:
            print("(no bird-eye-view image returned; matplotlib unavailable?)")

    print(f"tool_name={payload.get('tool_name')}, image_ids={result.get('image_ids')}")
    print("\n--- text_summary ---")
    print(text_summary)
    return payload


if __name__ == "__main__":
    print("== /v1/pose endpoint ==")
    test_server_pose_endpoint()
    print(f"\n== {TOOL_NAME} via router {TOOL_ROUTER_URL} ==")
    test_tool_via_router()
