# Crop_Tool Usage Guide

Crop an image using explicit box coordinates. Use this tool when you already know the target box and need the cropped region only. Do not use it to discover objects from text; use Visual_Grounding_Tool for region discovery first. Provide normalized coordinates in [0, 1].

## Parameters
- `image_path` (string, required): Runtime image ID from Environment Update messages (e.g., `input_image_1`, `tool_generated_image_1`).
- `coords` (array, required): Crop box as [x1, y1, x2, y2]. Normalized values in [0, 1] with x1 < x2 and y1 < y2.

## Call Format
{"type":"tool_call","name":"Crop_Tool","arguments":{"image_path":"input_image_1","coords":[0.1, 0.2, 0.5, 0.8]}}

## Best Practices
1. ALWAYS USE NORMALIZED COORDINATES — values in [0, 1].
2. ENSURE x1 < x2 AND y1 < y2 — the first pair is top-left, the second is bottom-right.
3. USE GROUNDING FIRST — if you don't already know the box coordinates, use Visual_Grounding_Tool to discover the region before cropping.
4. USE RUNTIME IMAGE IDs — pass image IDs from Environment Update messages (e.g., `input_image_1`), never invent file paths.

## Examples
### Example: Crop the top-left quarter of an image
{"type":"tool_call","name":"Crop_Tool","arguments":{"image_path":"input_image_1","coords":[0.0, 0.0, 0.5, 0.5]}}
Expected result: A dict with `cropped_image_base64` containing the cropped region.
