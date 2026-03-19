# Visual_Grounding_Tool Usage Guide

## Parameters
- `query` (string, required): Short noun phrase describing what to find in the image. Use singular form and simple visual descriptors.
- `image_path` (string, required): Runtime image ID from Environment Update messages (e.g., `input_image_1`, `tool_generated_image_1`).

## Call Format
{"type":"tool_call","name":"Visual_Grounding_Tool","arguments":{"query":"red car","image_path":"input_image_1"}}

## Best Practices
1. USE SINGULAR FORM: "tomato" not "tomatoes", "person" not "people"
2. USE SIMPLE VISUAL DESCRIPTORS: "red tomato" not "ripe tomato", "white car" not "expensive car"
3. NO TEXT/OCR QUERIES: Use "sign" not "stop sign text", use "license plate" not "ABC123"
4. KEEP PROMPTS CONCISE: "person" not "person who is walking across the street"
5. ITERATE IF NEEDED: If results are poor, try alternatives like "person" to "pedestrian" to "human"
6. USE RUNTIME IMAGE IDs — pass image IDs from Environment Update messages (e.g., `input_image_1`), never invent file paths.

## Examples
### Example: Locate a dog in an image
{"type":"tool_call","name":"Visual_Grounding_Tool","arguments":{"query":"dog","image_path":"input_image_1"}}
Expected result: Bounding boxes or masks highlighting detected dogs in the image.

### Example: Find red cars
{"type":"tool_call","name":"Visual_Grounding_Tool","arguments":{"query":"red car","image_path":"input_image_1"}}
Expected result: Bounding boxes or masks highlighting red cars in the image.
