# OCR_Tool Usage Guide

Extract readable text from an image using PaddleOCR-VL (a vision-language OCR model). Use this tool when the task is about words, numbers, or printed labels in the image. Do not use it for general object localization or segmentation.

## Parameters
- `image_id` (string, optional in the public schema): Runtime image ID from Environment Update messages (e.g., `input_image_1`, `tool_generated_image_1`).

Supply an available image ID in normal calls. Omitting the field does not
automatically select an input image; reference resolution only rewrites supplied fields.

## Call Format
{"type":"tool_call","name":"OCR_Tool","arguments":{"image_id":"input_image_1"}}

## Best Practices
1. USE FOR TEXT READING — this tool is specifically for extracting words, numbers, labels, signs, and any printed or handwritten text visible in the image. The underlying VLM handles diverse layouts and mixed scripts well.
2. DO NOT USE FOR OBJECT DETECTION — if you need to locate or identify non-text objects, use Visual_Grounding_Tool or segmentation tools instead.
3. CROP FIRST WHEN TEXT IS A SMALL FRACTION OF THE IMAGE — if the relevant text occupies only a tiny region, crop or zoom into that region first, then run OCR on the cropped image. For full-frame text the crop is usually unnecessary.
4. USE RUNTIME IMAGE IDs — pass image IDs from Environment Update messages (e.g., `input_image_1`), never invent file paths.

## Examples
### Example: Read text from a photo of a sign
{"type":"tool_call","name":"OCR_Tool","arguments":{"image_id":"input_image_1"}}
Expected result: `{"text": "STOP\nONE WAY", "model_id": "PaddlePaddle/PaddleOCR-VL-1.5"}`

### Example: Read text from a tool-generated cropped image
{"type":"tool_call","name":"OCR_Tool","arguments":{"image_id":"tool_generated_image_1"}}
Expected result: `{"text": "Hello World", "model_id": "PaddlePaddle/PaddleOCR-VL-1.5"}`
