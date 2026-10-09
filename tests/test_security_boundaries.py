"""Router inputs cannot name router-host files; configuration errors list every missing variable."""
import base64
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from openvlharness.bootstrap import load_agent_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "agent"
PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def test_router_rejects_local_image_paths_and_still_accepts_base64(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from openvlharness.services.server import create_app
    from test_services import config_for

    image = tmp_path / "private.png"
    Image.new("RGB", (64, 48), "white").save(image)
    with TestClient(create_app(config_for("zoom"))) as client:
        def zoom(**args):
            return client.post("/execute", json={"tool_name": "Zoom_In_Tool", "tool_call_id": "call_zoom",
                                                 "args": {"coords": [0, 0, 1000, 1000], **args}})
        for key, value in (("image_path", str(image)), ("image_paths", [str(image)])):
            response = zoom(**{key: value})
            assert response.status_code == 400
            assert response.json()["error"]["message"] == (
                f"Runtime/service fields cannot be supplied as task arguments: ['{key}']")
        response = zoom(image_base64=base64.b64encode(image.read_bytes()).decode())
        assert response.status_code == 200
        assert response.json()["result"]["cropped_image_base64"]


def test_missing_environment_variables_are_reported_in_one_error():
    path = CONFIGS / "gpt6_luna_paper.yaml"
    names = set(PLACEHOLDER.findall(path.read_text()))
    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError) as caught:
        load_agent_config(path)
    message = str(caught.value)
    assert message.startswith(f"Missing environment variables required by agent configuration {path}: ")
    assert "\n" not in message
    assert set(re.findall(r"\b([A-Z][A-Z0-9_]*) \(", message)) == names
    assert "TOOL_SERVER_URL (services.client.base_url)" in message
    assert ("DEPTH_ESTIMATION_SERVER_URL (services.tools.depth_estimation.server_url, "
            "services.tools.camera_trajectory.server_url)") in message


def test_router_cli_reports_configuration_errors_without_a_traceback(tmp_path):
    pytest.importorskip("fastapi")
    from openvlharness.services import server

    for config, expected in ((CONFIGS / "qwen3_8b_paper.yaml", "error: Missing environment variables"),
                             (tmp_path / "absent.yaml", "error: [Errno 2] No such file or directory")):
        argv = ["openvlharness.services.server", "--config", str(config)]
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", argv), \
                pytest.raises(SystemExit) as caught:
            server.main()
        assert str(caught.value.code).startswith(expected)
