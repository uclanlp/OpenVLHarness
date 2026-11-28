import re

from openvlharness.toolsmith import GeneratedTool, ToolCallError

GENERIC = {"Bridge", "Fireworks", "Festival", "Bay", "River", "Lake", "Harbor", "Harbour", "City", "Night", "View",
           "Skyline", "Port", "Beach", "Island", "Park", "Show", "Street", "Avenue", "Road", "Tower", "Photo",
           "Stock", "Image", "Images", "Pictures", "Wikipedia", "The", "Of", "In", "And", "At", "A"}
SETTLEMENTS = {"city", "town", "village", "municipality"}


def candidates(matches):
    counts = {}
    for match in matches:
        for phrase in re.findall(r"\b[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*", match.get("title", "")):
            if not set(phrase.split()) <= GENERIC:
                counts[phrase] = counts.get(phrase, 0) + 1
    return sorted(counts, key=lambda phrase: -counts[phrase])


class ImageScenicLocationIdentifier(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="image_scenic_location_identifier",
            tool_description=(
                "Identify the city or place shown in a scenic image, such as a skyline, bridge, harbor or festival "
                "scene, from reverse image search resolved with a gazetteer. Returns a place name. "
                "DO NOT USE FOR: people, animals, products, artwork, logos, or text-only questions."),
            input_schema={"type": "object", "properties": {
                "image_id": {"type": "string", "description": "Scenic image to locate."}},
                "required": ["image_id"]})

    def execute(self, image_base64=None):
        if not image_base64:
            return {"error": "an image is required"}
        try:
            matches = self.call_tool("visual_matches", image_base64=image_base64)
        except ToolCallError as exc:
            return {"error": str(exc)}
        matches = matches[:8] if isinstance(matches, list) else []
        best, score = None, 0.4
        for name in candidates(matches)[:4]:
            try:
                place = self.call_tool("gazetteer_lookup", query=name)
            except ToolCallError:
                continue
            value = (place.get("importance") or 0) + (0.2 if place.get("type") in SETTLEMENTS else 0)
            if place.get("found") and value > score:
                best, score = place, value
        if best:
            address = best.get("address") or {}
            return next((address[key] for key in ("city", "town", "village", "municipality") if address.get(key)),
                        best.get("display_name", "").split(",")[0])
        wiki = next((m for m in matches if "wikipedia.org/wiki/" in m.get("link", "")
                     and not m.get("title", "").startswith("File:")), None)
        if wiki is None:
            return {"error": "no candidate place could be resolved"}
        entity = re.split(r"\s+[-|–—·:]\s+", wiki.get("title", ""))[0]
        try:
            record = self.call_tool("wikidata_query", entity=entity,
                                    property="located in the administrative territorial entity")
        except ToolCallError as exc:
            return {"error": str(exc)}
        matched = record.get("matched") or {}
        if matched.get("confidence", 0) >= 1.0 and matched.get("values"):
            return matched["values"][0]
        return {"error": f"the location of {entity} is not in Wikidata"}
