"""Pure-CPU tool backends: zoom cropping and camera-motion interpretation."""
import base64
import io
import math

import numpy as np
import pytest
from PIL import Image

from openvlharness.services.tools.base import ToolArgumentError
from openvlharness.services.tools.camera_trajectory.interpret import interpret_camera_trajectory
from openvlharness.services.tools.zoom_in_image_by_bbox.tool import Zoom_In_Tool


def png_base64(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def crop_size(result):
    return Image.open(io.BytesIO(base64.b64decode(result["cropped_image_base64"]))).size


def test_zoom_pads_on_the_0_1000_scale_and_clips_to_the_image():
    tool = Zoom_In_Tool()
    image = png_base64(1000, 500)
    assert crop_size(tool.execute(image_base64=image, coords=[400, 400, 600, 600])) == (300, 150)
    assert crop_size(tool.execute(image_base64=image, coords=[0, 0, 100, 100], padding=100)) == (200, 100)
    with pytest.raises(ToolArgumentError, match="padding must be at least 50"):
        tool.execute(image_base64=image, coords=[0, 0, 10, 10], padding=10)
    with pytest.raises(ToolArgumentError, match="four values"):
        tool.execute(image_base64=image, coords=[0, 0, 10])


def w2c(yaw_deg=0.0, center=(0.0, 0.0, 0.0)):
    t = math.radians(yaw_deg)
    rotation = np.array([[math.cos(t), 0, -math.sin(t)], [0, 1, 0], [math.sin(t), 0, math.cos(t)]])
    return np.hstack([rotation, (-rotation @ np.asarray(center))[:, None]])


def test_camera_motion_is_expressed_in_the_first_camera_frame():
    first, moved, turned = interpret_camera_trajectory(
        np.stack([w2c(), w2c(center=(1.5, 0, 0)), w2c(yaw_deg=30)]))
    assert first["movement_summary"] == "reference camera"
    assert moved["camera_center_delta"] == pytest.approx([1.5, 0, 0], abs=1e-5)
    assert moved["movement_summary"].startswith("salient direction: right (+1.50)")
    assert turned["yaw_deg"] == pytest.approx(30, abs=1e-3)
    assert turned["rotation_summary"].startswith("salient direction: right (yaw +30.0°)")
    with pytest.raises(ValueError, match=r"\(N, 3, 4\)"):
        interpret_camera_trajectory(np.zeros((2, 4, 4)))
