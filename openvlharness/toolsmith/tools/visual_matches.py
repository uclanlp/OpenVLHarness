from openvlharness.services.tools.image_upload_utils import upload_image_bytes
from openvlharness.services.tools.image_utils import decode_base64_image_data, infer_image_content_type_and_extension
from openvlharness.services.tools.serper_common import SerperClient
from openvlharness.toolsmith import GeneratedTool


class VisualMatches(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="visual_matches",
            tool_description=(
                "Reverse image search that returns the raw Google Lens matches of an image, most similar first, as "
                "[{title, link, source, thumbnail, position}], for tools that rank or filter the sources themselves. "
                "DO NOT USE FOR: text queries or reading text in an image."),
            input_schema={"type": "object", "properties": {
                "image_id": {"type": "string", "description": "Image to search for."}},
                "required": ["image_id"]})

    def execute(self, image_base64=None):
        if not image_base64:
            return {"error": "an image is required"}
        data, _ = decode_base64_image_data(image_base64)
        content_type, extension = infer_image_content_type_and_extension(data)
        results = SerperClient().lens(upload_image_bytes(data, content_type, extension)).get("organic") or []
        matches = []
        for item in results:
            link = item.get("link") or ""
            if (item.get("title") or link) and all(link != match["link"] for match in matches):
                matches.append({"title": item.get("title") or "", "link": link, "source": item.get("source") or "",
                                "thumbnail": item.get("imageUrl") or item.get("thumbnailUrl") or "",
                                "position": len(matches) + 1})
        return matches[:10]
