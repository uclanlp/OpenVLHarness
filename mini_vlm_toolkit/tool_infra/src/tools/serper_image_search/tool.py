"""Image Search Tool — reverse image search via Serper Google Lens.

Input an image (public URL, local path, or base64) and get visual matches
including titles, links, sources, and thumbnails. Uses Serper's Google Lens
endpoint for high-quality object, landmark, artwork, and text recognition.
"""

import logging
import os
from typing import Dict, List, Optional
from urllib.parse import urlparse

from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from llm.engine import LLMEngine
from tools.base import BaseTool
from tools.image_utils import (
    decode_base64_image_data,
    infer_image_content_type_and_extension,
)
from tools.serper_common import SerperClient, get_serper_api_key, strip_think_tags

logger = logging.getLogger(__name__)

# LLM prompt to organize visual match results into coherent text.
IMAGE_SEARCH_SUMMARIZE_PROMPT = """You are a visual analysis assistant. Organize the following reverse image search (Google Lens) results into a clear, well-structured summary. Your goal is to help the user understand what the image depicts and what information is available about it.

## Visual Search Results
{raw_results}

## Instructions
1. Identify the most likely subject of the image based on the visual matches.
2. Group related matches (e.g., multiple results pointing to the same entity).
3. Summarize key findings: what the image shows, relevant context, and notable sources.
4. Include citation markers [1], [2], ... referencing specific results.
5. End with a Citations section listing source titles, source/page URLs, AND the direct image URLs.
6. **Preserve the trailing ` <img>` marker** that appears after each image URL in the raw results EXACTLY as-is. This marker is a framework signal used by downstream tools (e.g., Image_Search_Tool, ImageStateMemory) to register the image — dropping the marker breaks image-URL routing. Always emit it verbatim after the image URL.
7. Do NOT hallucinate — only describe what the search results indicate.

## Output Format
<summary of what the image likely depicts, with inline citations [1], [2], ...>

Citations:
[1] <title> | Image: <image_url> <img> | Source: <source_page_url>
[2] <title> | Image: <image_url> <img> | Source: <source_page_url>
...
"""

TOOL_NAME = "Image_Search_Tool"
DEFAULT_NUM_RESULTS = 10

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_url": {
            "type": "string",
            "description": (
                "A publicly accessible HTTP(S) URL pointing to the image to search. "
                "The image will be sent to Google Lens for visual recognition and "
                "reverse image search. Supported formats: JPEG, PNG, WebP, GIF, BMP."
            ),
        },
        "image_path": {
            "type": "string",
            "description": (
                "Absolute path to a local image file. The image will be automatically "
                "uploaded to cloud storage to obtain a public URL before searching. "
                "Use this when the image is stored locally on the server."
            ),
        },
        "image_base64": {
            "type": "string",
            "description": (
                "Base64-encoded image data (with or without data URI prefix). "
                "The image will be decoded, uploaded to cloud storage, and then "
                "searched. Use this when the image is passed inline as encoded data."
            ),
        },
    },
    "oneOf": [
        {"required": ["image_url"]},
        {"required": ["image_path"]},
        {"required": ["image_base64"]},
    ],
    "additionalProperties": False,
}


class Image_Search_Tool(BaseTool):
    """Reverse image search via Serper Google Lens with LLM-organized output.

    Identifies objects, landmarks, artwork, text, products, plants, animals,
    and more in an image. Returns an LLM-organized summary of visual matches
    with citations, or raw results if summarization is disabled.
    """

    require_llm_engine = True

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Identify objects, landmarks, artwork, text, products, plants, "
                "animals, and more in an image using Google Lens reverse image "
                "search. Returns a list of visual matches with titles, source URLs, "
                "links, and thumbnails.\n\n"
                "Use this tool when you need to understand what is depicted in an "
                "image — it can recognize specific places, products, artworks, "
                "logos, plants, animals, celebrities, etc. Provide the image as a "
                "public URL, a local file path, or base64-encoded data.\n\n"
                "This tool performs REVERSE image search (image → information). "
                "To search for images by text description, use Text_Search_Tool "
                "with search_type='images' instead."
            ),
            tool_version="1.0.0",
            input_types={
                "image_url": "str - public HTTP(S) URL of the image to search",
                "image_path": "str - absolute local image path (auto-uploaded to cloud storage)",
                "image_base64": "str - base64-encoded image data (auto-uploaded to cloud storage)",
            },
            input_schema=INPUT_SCHEMA,
            tool_aliases=["reverse_image_search", "visual_search"],
            output_type=(
                "list[dict] - list of visual matches, each with keys:\n"
                "  'title' (str): description or name of the matched content\n"
                "  'link' (str): URL to the source page\n"
                "  'source' (str): domain name of the source\n"
                "  'thumbnail' (str): URL to a thumbnail image\n"
                "  'position' (int): rank position in results"
            ),
            demo_commands=[
                {
                    "command": (
                        'Image_Search_Tool(image_url="https://example.com/landmark.jpg")'
                    ),
                    "description": (
                        "Identify a landmark in a publicly accessible image. "
                        "Returns matching places, Wikipedia entries, etc."
                    ),
                },
                {
                    "command": (
                        'Image_Search_Tool(image_path="/path/to/local/product.jpg")'
                    ),
                    "description": (
                        "Identify a product from a local image file. The image "
                        "is auto-uploaded to cloud storage before searching."
                    ),
                },
                {
                    "command": (
                        'Image_Search_Tool(image_base64="iVBORw0KGgo...")'
                    ),
                    "description": (
                        "Search using base64-encoded image data. Useful when "
                        "the image is passed inline from another tool's output."
                    ),
                },
            ],
            user_metadata={
                "api_key_env": "SERPER_API_KEY",
                "endpoint": "https://google.serper.dev/lens",
                "default_num_results": DEFAULT_NUM_RESULTS,
                "notes": (
                    "Uses Serper Google Lens API. Requires images to be "
                    "publicly accessible URLs. Local images and base64 data "
                    "are auto-uploaded via image_upload_utils."
                ),
            },
        )
        # LLM engine for result summarization
        model = os.getenv("SEARCH_LLM_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL
        base_url = os.getenv("SEARCH_LLM_BASE_URL") or os.getenv("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL
        api_key = os.getenv("SEARCH_LLM_API_KEY") or os.getenv("LLM_API_KEY") or DEFAULT_LLM_API_KEY
        _enable_thinking = os.getenv("SEARCH_LLM_ENABLE_THINKING", "false").strip().lower() in ("true", "1", "yes")
        self.llm_engine = LLMEngine(
            model=model, base_url=base_url, api_key=api_key,
            temperature=0.0, top_p=1.0,
            enable_thinking=_enable_thinking,
        )
        self._llm_summarize = os.getenv("SEARCH_LLM_SUMMARIZE", "true").strip().lower() in ("true", "1", "yes")
        self._llm_max_tokens = int(os.getenv("SEARCH_LLM_MAX_TOKENS", "4096"))
        self._num_results = int(os.getenv("SEARCH_NUM_RESULTS", "10"))
        self._client: Optional[SerperClient] = None

    def _get_client(self) -> SerperClient:
        if self._client is None:
            self._client = SerperClient()
        return self._client

    def sanity_check(self):
        try:
            get_serper_api_key()
        except Exception as exc:
            return [str(exc)]
        return []

    @staticmethod
    def _validate_image_url(image_url: str) -> str:
        """Validate that the provided string is a valid public HTTP(S) URL."""
        raw_url = str(image_url or "").strip()
        if not raw_url:
            raise ValueError("image_url is required")
        parsed = urlparse(raw_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("image_url must be a valid public http(s) URL")
        return raw_url

    def _resolve_image_url(
        self,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> str:
        """Resolve any input form to a public URL. Upload to cloud storage if needed."""
        if image_url:
            return self._validate_image_url(image_url)

        from tools.image_upload_utils import upload_image_bytes, upload_image_file

        if image_path:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"image_path not found: {image_path}")
            return upload_image_file(image_path)

        if image_base64:
            image_bytes, content_type = decode_base64_image_data(image_base64)
            if content_type:
                ext_map = {
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "image/webp": ".webp",
                    "image/gif": ".gif",
                    "image/bmp": ".bmp",
                    "image/tiff": ".tiff",
                }
                ext = ext_map.get(content_type.lower())
                if not ext:
                    content_type, ext = infer_image_content_type_and_extension(image_bytes)
            else:
                content_type, ext = infer_image_content_type_and_extension(image_bytes)
            return upload_image_bytes(image_bytes, content_type, ext)

        raise ValueError("Provide exactly one of: image_url, image_path, or image_base64")

    @staticmethod
    def _extract_item(item_data: Dict[str, object]) -> Optional[Dict[str, object]]:
        """Extract a visual match from a raw result item."""
        title = item_data.get("title")
        link = item_data.get("link")
        if not title and not link:
            return None
        return {
            "title": title or "",
            "link": link or "",
            "source": item_data.get("source", ""),
            "thumbnail": item_data.get("thumbnail", ""),
            "position": item_data.get("position", ""),
        }

    def _format_raw_results(self, results: List[Dict[str, object]]) -> str:
        """Format raw visual matches into numbered text for LLM or direct output."""
        lines = []
        for i, item in enumerate(results, 1):
            lines.append(f"{i}. {item.get('title', '')}")
            if item.get("link"):
                lines.append(f"   URL: {item['link']}")
            if item.get("source"):
                lines.append(f"   Source: {item['source']}")
        return "\n".join(lines)

    def _summarize_with_llm(self, raw_text: str) -> str:
        """Pass raw visual match results through LLM for organized output."""
        prompt = IMAGE_SEARCH_SUMMARIZE_PROMPT.format(raw_results=raw_text)
        try:
            result = self.llm_engine.generate(
                prompt, max_tokens=self._llm_max_tokens, temperature=0.0, top_p=1.0,
            )
            stripped = strip_think_tags(result or "")
            # Empty / whitespace-only LLM output is not a useful summary —
            # fall back to raw_text so the agent still gets the underlying
            # citations + ``<img>`` sentinels intact.
            if not stripped.strip():
                logger.warning(
                    "LLM summarization returned empty content, returning raw results"
                )
                return raw_text
            return stripped
        except Exception as exc:
            logger.warning("LLM summarization failed (%s), returning raw results", exc)
            return raw_text

    def execute(
        self,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> str:
        """Execute reverse image search via Google Lens.

        Args:
            image_url: Public HTTP(S) URL of the image.
            image_path: Local file path (auto-uploaded).
            image_base64: Base64-encoded image data (auto-uploaded).

        Returns:
            LLM-organized summary of visual matches with citations,
            or raw formatted results if summarization is disabled.
        """
        resolved_url = self._resolve_image_url(image_url, image_path, image_base64)
        client = self._get_client()
        payload = client.lens(resolved_url)

        raw_items: List[Dict[str, object]] = payload.get("organic", [])

        results = []
        seen = set()
        for item_data in raw_items:
            if not isinstance(item_data, dict):
                continue
            match = self._extract_item(item_data)
            if match is None:
                continue
            dedup_key = match["link"] or match["title"]
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            results.append(match)
            if len(results) >= self._num_results:
                break

        if not results:
            return "[Image Search] No visual matches found for this image."

        raw_text = self._format_raw_results(results)

        if self._llm_summarize:
            summary = self._summarize_with_llm(raw_text)
            return f"[Image Search — Visual Matches]\n{summary}"

        return f"[Image Search — Visual Matches]\n{raw_text}"
