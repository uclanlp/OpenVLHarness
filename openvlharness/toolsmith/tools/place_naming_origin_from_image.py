import re

from openvlharness.toolsmith import GeneratedTool, ToolCallError

AUTHORITY = (("wikipedia.org", 100), ("wikidata.org", 90), ("britannica.com", 80), (".gov", 75),
             ("official", 70))


def authority(match):
    text = f"{match.get('link', '')} {match.get('source', '')}".lower()
    return max((weight for marker, weight in AUTHORITY if marker in text), default=0)


def name_of(title):
    return re.split(r"\s+[-|–—·:]\s+", title or "")[0].strip()


def decoded(escapes):
    return bytes.fromhex(escapes.group(0).replace("%", "")).decode("utf-8", "replace")


def article(text):
    found = re.search(r"wikipedia\.org/wiki/([^\s?#/]+)", text or "")
    return re.sub(r"(?:%[0-9A-Fa-f]{2})+", decoded, found.group(1)).replace("_", " ") if found else ""


class PlaceNamingOriginFromImage(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="place_naming_origin_from_image",
            tool_description=(
                "Identify a square, street, building or other named place from an image by reverse image search, then "
                "report whom or what its name comes from, from Wikidata. DO NOT USE FOR: people, animals, plants, "
                "products, artwork, or text-only questions."),
            input_schema={"type": "object", "properties": {
                "image_id": {"type": "string", "description": "Image of the place."}},
                "required": ["image_id"]})

    def identify(self, image_base64):
        """Name the subject from a Wikipedia match, else from the article a web search finds for the best match."""
        matches = self.call_tool("visual_matches", image_base64=image_base64)
        if not isinstance(matches, list):
            return ""
        matches = [match for match in matches[:8] if not match.get("title", "").startswith("File:")]
        if not matches:
            return ""
        best = max(matches, key=authority)
        if authority(best) >= 100:
            return article(best["link"]) or name_of(best["title"])
        found = self.call_tool("Text_Search_Tool", query=f"{name_of(best['title'])} wikipedia")
        return article(found.get("text", "") if isinstance(found, dict) else found) or name_of(best["title"])

    def execute(self, image_base64=None):
        if not image_base64:
            return {"error": "an image is required"}
        try:
            entity = self.identify(image_base64)
            if not entity:
                return {"error": "no visual match names a place"}
            record = self.call_tool("wikidata_query", entity=entity, property="named after")
        except ToolCallError as exc:
            return {"error": str(exc)}
        matched = record.get("matched") or {}
        if not record.get("found") or matched.get("confidence", 0) < 1.0 or not matched.get("values"):
            return {"error": f"Wikidata gives no name origin for {entity}", "candidate": entity}
        return ", ".join(matched["values"][:3])
