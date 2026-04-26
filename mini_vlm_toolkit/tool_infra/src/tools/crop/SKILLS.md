# Crop_Tool Usage Guide

Crop an image using explicit box coordinates. Use this tool when you already know the target box and need the cropped region only. Do not use it to discover objects from text; use Visual_Grounding_Tool for region discovery first. Provide normalized coordinates on the 0-1000 scale (each value in [0, 1000]).

## Parameters
- `image_path` (string, required): Runtime image ID from Environment Update messages (e.g., `input_image_1`, `tool_generated_image_1`).
- `coords` (array, required): Crop box as [x1, y1, x2, y2]. Normalized values on the 0-1000 scale with x1 < x2 and y1 < y2.

## Call Format
{"type":"tool_call","name":"Crop_Tool","arguments":{"image_path":"input_image_1","coords":[100, 200, 500, 800]}}

## Best Practices
1. ALWAYS USE 0-1000 NORMALIZED COORDINATES — values in [0, 1000].
2. ENSURE x1 < x2 AND y1 < y2 — the first pair is top-left, the second is bottom-right.
3. USE GROUNDING FIRST — if you don't already know the box coordinates, use Visual_Grounding_Tool to discover the region before cropping.
4. USE RUNTIME IMAGE IDs — pass image IDs from Environment Update messages (e.g., `input_image_1`), never invent file paths.

## Examples
### Example: Crop the top-left quarter of an image
{"type":"tool_call","name":"Crop_Tool","arguments":{"image_path":"input_image_1","coords":[0, 0, 500, 500]}}
Expected result: A dict with `cropped_image_base64` containing the cropped region.
