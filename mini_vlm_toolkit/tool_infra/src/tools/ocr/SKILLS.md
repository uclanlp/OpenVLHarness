# OCR_Tool Usage Guide

Extract readable text from an image using an OCR server. Use this tool when the task is about words, numbers, or printed labels in the image. Do not use it for general object localization or segmentation.

## Parameters
- `image_path` (string, required): Runtime image ID from Environment Update messages (e.g., `input_image_1`, `tool_generated_image_1`).

## Call Format
{"type":"tool_call","name":"OCR_Tool","arguments":{"image_path":"input_image_1"}}

## Best Practices
1. USE FOR TEXT READING — this tool is specifically for extracting words, numbers, labels, signs, and any printed or handwritten text visible in the image.
2. DO NOT USE FOR OBJECT DETECTION — if you need to locate or identify non-text objects, use Visual_Grounding_Tool or segmentation tools instead.
3. CROP FIRST IF NEEDED — if the text region is small or the image is cluttered, crop or zoom into the relevant area first, then run OCR on the cropped image for better accuracy.
4. USE RUNTIME IMAGE IDs — pass image IDs from Environment Update messages (e.g., `input_image_1`), never invent file paths.

## Examples
### Example: Read text from a photo of a sign
{"type":"tool_call","name":"OCR_Tool","arguments":{"image_path":"input_image_1"}}
Expected result: `{"texts": ["STOP", "ONE WAY"], "results": [...], "languages": ["en"], "gpu": true}`

### Example: Read text from a tool-generated cropped image
{"type":"tool_call","name":"OCR_Tool","arguments":{"image_path":"tool_generated_image_1"}}
Expected result: `{"texts": ["Hello World"], "results": [...], "languages": ["en"], "gpu": true}`
