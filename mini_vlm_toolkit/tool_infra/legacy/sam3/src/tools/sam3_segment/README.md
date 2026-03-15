# SAM3 Segment Tool

This tool calls an external SAM3 segmentation server. The server code is copied
from `reference/tool-vision/servers/sam3_server.py` for convenience.

## Running the SAM3 server

```bash
./src/tools/sam3_segment/run_sam3_server.sh
```

Environment variables (optional):
- `HOST` (default: `127.0.0.1`)
- `PORT` (default: `8083`)
- Set the device externally (for example, via `CUDA_VISIBLE_DEVICES`) before running the script.

## SAM3 server dependencies

You need a Python environment that can import and run the SAM3 server:

1) Install `torch` separately to match your system/accelerator.
2) Install server dependencies (includes SAM3 from source):
   - `pip install -r requirements-server.txt`
3) Request access to `facebook/sam3` on huggingface.
4) If you download SAM3 from source, make sure `PYTHONPATH` includes the SAM3 package path.

The toolkit itself only needs `requests` to call the server.

## Image inputs

`Visual_Grounding_Tool` accepts either `image_path` (server-local path) or
`image_base64` (base64-encoded image data, optionally a data URL). Provide
exactly one of the two.

```python
import base64

with open("tests/data/sam3_demo.jpg", "rb") as handle:
    image_base64 = base64.b64encode(handle.read()).decode("utf-8")

tool.execute(prompt="shoe", image_base64=image_base64)
```

## Output visualization

`Visual_Grounding_Tool` returns `all_masks_viz_base64` by default. Set
`return_all_masks_viz=False` to skip generating the visualization.
If you set `return_masks_png_b64=False` while keeping the visualization enabled,
the tool still requests masks from the server and removes them from the final response.
