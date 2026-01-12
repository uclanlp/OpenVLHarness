import base64
import os
import urllib.request
from io import BytesIO
from datetime import datetime

import requests
from PIL import Image


DEFAULT_VISION_TEST_IMAGE_URL = (
    "https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/hopper.jpg"
)
DEFAULT_OVERLAY_TEST_IMAGE_URL = (
    "https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/pil123rgba.png"
)
DEFAULT_OCR_TEST_IMAGE_URL = (
    "https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/english.png"
)
DEFAULT_DEPTH_TEST_IMAGE_URL = DEFAULT_VISION_TEST_IMAGE_URL


def ensure_tool_server(server_url, required_tools=None, timeout_seconds=5):
    if not server_url:
        print("TOOL_SERVER_URL is not set; skipping tests.")
        return False
    endpoint = f"{server_url.rstrip('/')}/tools"
    try:
        response = requests.get(endpoint, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"Tool server unavailable at {endpoint}: {exc}")
        return False
    if not isinstance(payload, dict):
        print(f"Tool server returned unexpected payload from {endpoint}")
        return False
    missing = [tool for tool in (required_tools or []) if tool not in payload]
    if missing:
        missing_list = ", ".join(missing)
        print(f"Tool server missing required tools: {missing_list}. Skipping tests.")
        return False
    return True


def ensure_sam3_server(server_url, timeout_seconds=5):
    if not server_url:
        print("SAM3_SERVER_URL is not set; skipping SAM3 tests.")
        return False
    endpoint = f"{server_url.rstrip('/')}/health"
    try:
        response = requests.get(endpoint, timeout=timeout_seconds)
    except Exception as exc:
        print(f"SAM3 server unavailable at {endpoint}: {exc}")
        return False
    if response.status_code != 200:
        print(
            f"SAM3 server unhealthy at {endpoint}: status {response.status_code}"
        )
        return False
    return True


def ensure_grounding_dino_server(server_url, timeout_seconds=5):
    if not server_url:
        print("GROUNDING_DINO_SERVER_URL is not set; skipping GroundingDINO tests.")
        return False
    endpoint = f"{server_url.rstrip('/')}/health"
    try:
        response = requests.get(endpoint, timeout=timeout_seconds)
    except Exception as exc:
        print(f"GroundingDINO server unavailable at {endpoint}: {exc}")
        return False
    if response.status_code != 200:
        print(
            f"GroundingDINO server unhealthy at {endpoint}: status {response.status_code}"
        )
        return False
    return True


def ensure_ocr_server(server_url, timeout_seconds=5):
    if not server_url:
        print("OCR_SERVER_URL is not set; skipping OCR tests.")
        return False
    endpoint = f"{server_url.rstrip('/')}/health"
    try:
        response = requests.get(endpoint, timeout=timeout_seconds)
    except Exception as exc:
        print(f"OCR server unavailable at {endpoint}: {exc}")
        return False
    if response.status_code != 200:
        print(f"OCR server unhealthy at {endpoint}: status {response.status_code}")
        return False
    return True


def ensure_depth_server(server_url, timeout_seconds=5):
    if not server_url:
        print("DEPTH_ANYTHING_SERVER_URL is not set; skipping depth tests.")
        return False
    endpoint = f"{server_url.rstrip('/')}/health"
    try:
        response = requests.get(endpoint, timeout=timeout_seconds)
    except Exception as exc:
        print(f"Depth server unavailable at {endpoint}: {exc}")
        return False
    if response.status_code != 200:
        print(f"Depth server unhealthy at {endpoint}: status {response.status_code}")
        return False
    return True


def ensure_test_image(
    env_var="VISION_TEST_IMAGE",
    url_env_var="VISION_TEST_IMAGE_URL",
    default_url=DEFAULT_VISION_TEST_IMAGE_URL,
    filename="vision_tools_demo.jpg",
):
    image_path = os.getenv(env_var)
    if image_path and os.path.isfile(image_path):
        return image_path

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    image_path = os.path.join(data_dir, filename)
    if os.path.isfile(image_path):
        return image_path

    image_url = os.getenv(url_env_var, default_url)
    print(f"Downloading test image from {image_url}")
    try:
        urllib.request.urlretrieve(image_url, image_path)
    except Exception as exc:
        print(f"Failed to download test image: {exc}")
        return None
    return image_path


def ensure_overlay_image():
    return ensure_test_image(
        env_var="OVERLAY_TEST_IMAGE",
        url_env_var="OVERLAY_TEST_IMAGE_URL",
        default_url=DEFAULT_OVERLAY_TEST_IMAGE_URL,
        filename="overlay_tools_demo.png",
    )


def ensure_ocr_test_image():
    return ensure_test_image(
        env_var="OCR_TEST_IMAGE",
        url_env_var="OCR_TEST_IMAGE_URL",
        default_url=DEFAULT_OCR_TEST_IMAGE_URL,
        filename="ocr_tools_demo.png",
    )


def ensure_depth_test_image():
    return ensure_test_image(
        env_var="DEPTH_TEST_IMAGE",
        url_env_var="DEPTH_TEST_IMAGE_URL",
        default_url=DEFAULT_DEPTH_TEST_IMAGE_URL,
        filename="depth_tools_demo.jpg",
    )


def decode_base64_image(image_base64):
    data = base64.b64decode(image_base64)
    return Image.open(BytesIO(data))


def ensure_blank_image(filename="blank_canvas.png", size=(200, 200), color=(255, 255, 255)):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    image_path = os.path.join(data_dir, filename)
    if os.path.isfile(image_path):
        return image_path
    image = Image.new("RGB", size, color)
    image.save(image_path, format="PNG")
    return image_path


def build_test_id(file_path, function_name):
    base = os.path.splitext(os.path.basename(file_path))[0]
    return f"{base}.{function_name}"


def save_test_output(image, test_id, suffix=None, image_format="PNG"):
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_test_id = _sanitize_filename_component(test_id)
    if suffix:
        safe_suffix = _sanitize_filename_component(str(suffix))
        filename = f"{safe_test_id}_{safe_suffix}_{timestamp}.png"
    else:
        filename = f"{safe_test_id}_{timestamp}.png"
    output_path = os.path.join(output_dir, filename)
    image.save(output_path, format=image_format)
    return output_path


def _sanitize_filename_component(value):
    sanitized = value.replace(os.sep, "_").replace(" ", "_")
    return sanitized.replace("/", "_")
