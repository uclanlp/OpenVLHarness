# Reverse image search

Use the shared [search setup](../serper_text_search/README.md) for the configured
Qwen multimodal summarizer, Serper credentials and S3/GCS image-upload setup.
`Image_Search_Tool` accepts one runtime `image_id`; the capability resolves it
before the router uploads the image for Serper Lens. The paper output is text
or text with retained image bytes, so `matches` is null unless structured
records were actually returned. Text-to-image search is a different mode:
`Text_Search_Tool(search_type="images")`.
