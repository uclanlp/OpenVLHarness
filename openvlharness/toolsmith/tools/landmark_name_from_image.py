import re

from openvlharness.toolsmith import GeneratedTool, ToolCallError

AUTHORITY = (("wikipedia.org", 100), ("wikidata.org", 90), ("britannica.com", 80), (".gov", 75),
             ("official", 70), ("unesco.org", 65))


def authority(match):
    text = f"{match.get('link', '')} {match.get('source', '')}".lower()
    return max((weight for marker, weight in AUTHORITY if marker in text), default=0)


def name_of(title):
    return re.split(r"\s+[-|–—·:]\s+", title or "")[0].strip()


def ranked(matches):
    if not isinstance(matches, list):
        return []
    articles = [match for match in matches[:8] if not match.get("title", "").startswith("File:")]
    return sorted(articles, key=authority, reverse=True)


class LandmarkNameFromImage(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="landmark_name_from_image",
            tool_description=(
                "Identify the proper name of a landmark or structure in an image, such as a building, bridge, tower, "
                "lighthouse or monument, by reverse image search with a Wikidata check. DO NOT USE FOR: people, "
                "animals, plants, products, logos, or text-only questions."),
            input_schema={"type": "object", "properties": {
                "image_id": {"type": "string", "description": "Image of the landmark."}},
                "required": ["image_id"]})

    def execute(self, image_base64=None):
        if not image_base64:
            return {"error": "an image is required"}
        try:
            candidates = ranked(self.call_tool("visual_matches", image_base64=image_base64))
            name = name_of(candidates[0]["title"]) if candidates else ""
            if not name:
                return {"error": "no visual match names a landmark"}
            if authority(candidates[0]) >= 80:
                return name
            record = self.call_tool("wikidata_query", entity=name, property="instance of")
        except ToolCallError as exc:
            return {"error": str(exc)}
        matched = record.get("matched") or {}
        if record.get("found") and matched.get("confidence", 0) >= 1.0 and matched.get("values"):
            return record.get("entity_label") or name
        return {"error": f"{name} is not confirmed by Wikidata", "candidate": name}
