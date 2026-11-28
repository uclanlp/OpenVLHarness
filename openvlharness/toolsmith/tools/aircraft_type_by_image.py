import re

from openvlharness.toolsmith import GeneratedTool, ToolCallError

AVIATION = ("wikipedia.org", "airliners.net", "jetphotos.com", "planespotters.net", "aviation")
FAMILY = re.compile(r"\b(Airbus|Boeing|Embraer|Bombardier|ATR|Cessna|Lockheed|McDonnell|Douglas|Antonov|Tupolev|"
                    r"Ilyushin|Sukhoi|Mikoyan|Dassault|Gulfstream|Beechcraft|Pilatus|de Havilland|Fokker|Saab|"
                    r"Northrop|Grumman|Piper|Comac|Mitsubishi)\b[\w\s/-]{0,24}?\b[A-Z]{0,3}-?\d{2,4}[A-Z\d-]*")


class AircraftTypeByImage(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="aircraft_type_by_image",
            tool_description=(
                "Identify the type or model of an aircraft in an image, such as an airliner, business jet or military "
                "plane, from the titles of its reverse image search matches. DO NOT USE FOR: cars, ships, people, "
                "animals, or images without an aircraft."),
            input_schema={"type": "object", "properties": {
                "image_id": {"type": "string", "description": "Image of the aircraft."}},
                "required": ["image_id"]})

    def execute(self, image_base64=None):
        if not image_base64:
            return {"error": "an image is required"}
        try:
            matches = self.call_tool("visual_matches", image_base64=image_base64)
        except ToolCallError as exc:
            return {"error": str(exc)}
        counts = {}
        for match in matches[:10] if isinstance(matches, list) else []:
            weight = 2 if any(site in match.get("link", "").lower() for site in AVIATION) else 1
            for found in FAMILY.finditer(match.get("title", "")):
                model = found.group(0).strip()
                counts[model] = counts.get(model, 0) + weight
        if not counts:
            return {"error": "no visual match names an aircraft model"}
        return max(counts, key=counts.get)
