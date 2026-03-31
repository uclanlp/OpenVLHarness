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
| `image_url` | str | one of three | Public HTTP(S) image URL |
| `image_path` | str | one of three | Local image path (auto-uploaded to cloud storage) |
| `image_base64` | str | one of three | Base64-encoded image data (auto-uploaded) |

Provide exactly **one** of the three image input parameters.

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

# Identify content in a local image
SerpApi_Google_Lens_Tool(image_path="input_image_1")
```

## API

Uses SerpApi's `google_lens` engine. Shares the same API key as `SerpApi_Image_Search_Tool`.
