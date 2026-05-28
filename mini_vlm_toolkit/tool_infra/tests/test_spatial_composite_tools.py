import base64
from contextlib import contextmanager

import requests

import numpy as np

from tools.spatial_utils import build_tracking_summary
from tools.tracking.tool import Tracking_Tool


PNG_STUB = "data:image/png;base64," + ("A" * 80)
IMG1 = "data:image/png;base64," + base64.b64encode(b"img1").decode("ascii")
IMG2 = "data:image/png;base64," + base64.b64encode(b"img2").decode("ascii")
IMG3 = "data:image/png;base64," + base64.b64encode(b"img3").decode("ascii")


class FakeResponse:
    ok = True
    status_code = 200
    reason = "OK"
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _depth_payload():
    depth = np.ones((100, 100), dtype=np.float32)
    return {
        "depth_meters_b64": base64.b64encode(depth.tobytes()).decode("ascii"),
        "depth_shape": [100, 100],
        "colormap": "inferno",
        "depth_map_base64": PNG_STUB,
    }


def _grounding_payload(query, image_key):
    boxes_by_query = {
        "red block": [[10, 40, 30, 60]],
        "blue bowl": [[70, 40, 90, 60]],
        "missing": [],
    }
    if query == "ball":
        if image_key == "img1":
            boxes = [[10, 40, 30, 60]]
        elif image_key == "img2":
            boxes = [[45, 40, 65, 60]]
        else:
            boxes = [[80, 40, 98, 58]]
    else:
        boxes = boxes_by_query.get(query, [[20, 20, 40, 40]])
    return {
        "prompt": query,
        "pred_boxes": boxes,
        "pred_scores": [0.9 for _ in boxes],
        "pred_masks": [],
        "num_masks": len(boxes),
        "orig_img_w": 100,
        "orig_img_h": 100,
        "rendered_image_b64": PNG_STUB,
    }


def fake_post(url, files=None, data=None, timeout=None):
    del timeout
    if url.endswith("/v1/depth"):
        return FakeResponse(_depth_payload())
    if url.endswith("/v1/segment"):
        image_bytes = (files or {})["image"][1]
        image_key = image_bytes.decode("utf-8", errors="ignore")
        return FakeResponse(_grounding_payload((data or {}).get("prompt"), image_key))
    raise AssertionError(f"Unexpected URL: {url}")


@contextmanager
def patched_requests_post():
    original_post = requests.post
    requests.post = fake_post
    try:
        yield
    finally:
        requests.post = original_post


def test_tracking_tool_tracks_rightward_motion():
    tool = Tracking_Tool()

    with patched_requests_post():
        result = tool.execute(
            target="ball",
            image_base64s=[IMG1, IMG2, IMG3],
            _image_ids=["input_image_1", "input_image_2", "input_image_3"],
        )

    assert result["status"] == "tracked"
    assert result["movement"]["dx"] > 0
    assert result["movement"]["magnitude_2d"] > 0
    assert len(result["tracks"]) == 3
    assert len(result["grounding_images"]) == 3
    summary = result["summary"]
    assert "right" in summary
    assert "image-normalized" in summary
    assert "Δx=" in summary
    assert "moved opposite" in summary.lower()


def test_tracking_tool_below_noise_floor_drops_direction_words():
    movement = {
        "dx": -0.002,
        "dy": +0.001,
        "magnitude_2d": 0.002,
        "start_depth": 0.250,
        "end_depth": 0.254,
        "depth_delta": +0.004,
    }
    summary = build_tracking_summary("gripper", movement)
    assert "moves left" not in summary
    assert "moves right" not in summary
    assert "moves up" not in summary
    assert "moves down" not in summary
    assert "does not move" not in summary
    assert "The gripper is tracked" in summary
    assert "Δx=-0.002" in summary
    assert "magnitude=0.002" in summary
    assert "0.250 m" in summary
    assert "moved opposite" in summary.lower()


def main():
    test_tracking_tool_tracks_rightward_motion()
    print("Spatial composite tool tests completed.")


if __name__ == "__main__":
    main()
