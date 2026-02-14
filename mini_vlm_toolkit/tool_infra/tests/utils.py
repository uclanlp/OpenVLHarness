import base64
import contextlib
import json
import mimetypes
import os
import threading
import urllib.request
from io import BytesIO
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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


@contextlib.contextmanager
def run_json_error_server(status_code=400, payload=None):
    payload = payload or {"detail": "test error"}
    encoded = json.dumps(payload).encode("utf-8")

    class ErrorHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            if length:
                self.rfile.read(length)
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), ErrorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def load_env_file(env_path=None, override=False):
    root_dir = os.path.dirname(os.path.dirname(__file__))
    env_path = env_path or os.path.join(root_dir, ".env")
    if not os.path.isfile(env_path):
        return False

    with open(env_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value and value[0] == value[-1] and value[0] in ("\"", "'"):
                value = value[1:-1]
            if override or key not in os.environ:
                os.environ[key] = value
    return True


def get_model_base_url_from_env(model_name, env_var="LLM_DIRECT_MODEL_BASE_URLS"):
    if not model_name:
        return None
    raw_value = os.getenv(env_var)
    if not raw_value:
        return None
    for item in raw_value.split(","):
        entry = item.strip()
        if not entry or ":" not in entry:
            continue
        name, base_url = entry.split(":", 1)
        name = name.strip()
        base_url = base_url.strip()
        if name == model_name and base_url:
            return base_url
    return None


def ensure_openai_compat_server(base_url, timeout_seconds=5):
    if not base_url:
        return False
    trimmed = base_url.rstrip("/")
    endpoint = f"{trimmed}/models" if trimmed.endswith("/v1") else f"{trimmed}/v1/models"
    try:
        response = requests.get(endpoint, timeout=timeout_seconds)
    except Exception as exc:
        print(f"LLM server unavailable at {endpoint}: {exc}")
        return False
    if response.status_code != 200:
        print(f"LLM server unhealthy at {endpoint}: status {response.status_code}")
        return False
    return True


def ensure_model_available(base_url, model_name, timeout_seconds=5):
    if not model_name:
        return False
    trimmed = base_url.rstrip("/")
    endpoint = f"{trimmed}/models" if trimmed.endswith("/v1") else f"{trimmed}/v1/models"
    try:
        response = requests.get(endpoint, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"LLM model listing unavailable at {endpoint}: {exc}")
        return False
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        print(f"Unexpected model list payload from {endpoint}")
        return False
    for entry in data:
        if isinstance(entry, dict) and entry.get("id") == model_name:
            return True
    print(f"Model {model_name} not found in {endpoint}")
    return False


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
        print("SAM3 server URL is not set; skipping SAM3 tests.")
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


def encode_image_base64(image_path, include_data_url=False):
    with open(image_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    if not include_data_url:
        return encoded
    content_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
    return f"data:{content_type};base64,{encoded}"


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
