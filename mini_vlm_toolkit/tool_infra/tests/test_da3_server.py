"""Quick test: POST an image to the DA3 depth estimation server and save results."""

import base64
import requests
import numpy as np
from pathlib import Path

IMAGE_PATH = Path(__file__).resolve().parent / "data" / "test_depth.png"
SERVER_URL = "http://127.0.0.1:8087"

with open(IMAGE_PATH, "rb") as f:
    resp = requests.post(
        f"{SERVER_URL}/v1/depth",
        files={"image": (IMAGE_PATH.name, f, "image/png")},
        data={"colormap": "inferno"},
        timeout=60,
    )

resp.raise_for_status()
result = resp.json()
print(result.keys())
# Save colorized depth visualization
depth_png_bytes = base64.b64decode(result["depth_map_base64"])
with open("depth_output.png", "wb") as f:
    f.write(depth_png_bytes)
print(f"Saved depth_output.png")

# Decode raw float32 depth array
shape = result["depth_shape"]
depth_array = np.frombuffer(
    base64.b64decode(result["depth_meters_b64"]), dtype=np.float32
).reshape(shape)
print(f"Depth array shape: {depth_array.shape}, range: [{depth_array.min():.4f}, {depth_array.max():.4f}]")
print(f"Colormap: {result['colormap']}, Model: {result['model_id']}")
