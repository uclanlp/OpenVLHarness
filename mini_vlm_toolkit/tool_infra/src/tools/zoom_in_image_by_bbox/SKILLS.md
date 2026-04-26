# Zoom_In_Tool Usage Guide

Zoom into a region by cropping an image with a normalized bbox [x1, y1, x2, y2] plus padding, and return the cropped image. Use this tool for closer visual inspection after you already know the region. All coords values must be normalized on a 0-1000 scale (each value in [0, 1000]). Do not use it to find regions from text prompts; use GroundingDINO_Tool or SAM tools for region discovery/segmentation first.

## Parameters
- `image_path` (string, required): Runtime image ID from Environment Update messages (e.g., `input_image_1`, `tool_generated_image_1`).
- `coords` (array, required): Normalized bbox [x1, y1, x2, y2] on the 0-1000 scale where every value must be in [0, 1000]. (x1, y1) is the top-left corner; (x2, y2) is the bottom-right corner.
- `padding` (number, optional): Extra normalized margin on each side on the 0-1000 scale (must be >= 50). Default: 50.

## Call Format
{"type":"tool_call","name":"Zoom_In_Tool","arguments":{"image_path":"input_image_1","coords":[200, 300, 600, 600],"padding":100}}

## Best Practices
1. USE 0-1000 COORDINATES ONLY — all four values in `coords` must be in [0, 1000]. This tool does not accept pixel coordinates or 0-1 normalized coordinates.
2. FORMAT IS [x1, y1, x2, y2] NOT [x, y, w, h] — (x1, y1) is top-left, (x2, y2) is bottom-right. This is the same xyxy format used by GroundingDINO_Tool and Crop_Tool.
3. USE PADDING FOR CONTEXT — the default padding of 50 adds a small margin. Increase it (e.g., 100 or 150) if you need more surrounding context. Use ZoomIn when you want surrounding context via padding; use Crop_Tool for exact crops without padding.
4. DISCOVER REGIONS FIRST — use a grounding or segmentation tool to find the bbox before zooming in.
5. USE RUNTIME IMAGE IDs — pass image IDs from Environment Update messages (e.g., `input_image_1`), never invent file paths.

## Examples
### Example: Zoom into a detected object region
{"type":"tool_call","name":"Zoom_In_Tool","arguments":{"image_path":"input_image_1","coords":[200, 300, 600, 600],"padding":50}}
Expected result: A dict with `cropped_image_base64` containing the zoomed-in region with padding.

### Example: Zoom with extra context padding
{"type":"tool_call","name":"Zoom_In_Tool","arguments":{"image_path":"input_image_1","coords":[500, 100, 700, 250],"padding":150}}
Expected result: A dict with `cropped_image_base64` containing the zoomed-in region with generous surrounding context.
