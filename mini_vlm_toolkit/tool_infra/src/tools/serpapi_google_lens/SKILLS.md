# SerpApi_Google_Lens_Tool

Identify objects, landmarks, artwork, text, and more in an image using Google Lens.

## When to use

- You need to identify **what** is in an image (a building, artwork, product, animal, etc.)
- You want to find the **name** of a place, object, or artwork shown in an image
- You need web pages that discuss the content depicted in an image

## When NOT to use

- You already know what's in the image and just need to search by text → use `Serper_Google_Search_Tool`
- You need to find visually similar images (not identify content) → use `SerpApi_Image_Search_Tool`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_url` | str | one of two | Public HTTP(S) image URL |
| `image_id` | str | one of two | Runtime image ID from Environment Update messages (e.g. `input_image_1`); auto-uploaded to cloud storage |

Provide exactly **one** of the two image input parameters.

## Output

Returns `list[dict]` with up to 10 visual matches. Each dict contains:

| Field | Description |
|-------|-------------|
| `title` | Title of the matching web page |
| `link` | URL of the matching web page |
| `source` | Source website name |
| `thumbnail` | Thumbnail image URL |
| `position` | Result position (rank) |

## Examples

```python
# Identify a landmark from a public URL
SerpApi_Google_Lens_Tool(image_url="https://example.com/building.jpg")

# Identify content in a runtime image
SerpApi_Google_Lens_Tool(image_id="input_image_1")
```

## API

Uses SerpApi's `google_lens` engine. Shares the same API key as `SerpApi_Image_Search_Tool`.
