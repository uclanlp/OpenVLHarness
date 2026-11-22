import requests

from openvlharness.toolsmith import GeneratedTool

HEADERS = {"User-Agent": "OpenVLHarness-ToolSmith/0.1 (https://github.com/uclanlp/OpenVLHarness)",
           "Accept": "application/json"}
SEARCH = "https://nominatim.openstreetmap.org/search"


class GazetteerLookup(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="gazetteer_lookup",
            tool_description=(
                "Geocode a place name with OpenStreetMap Nominatim. Returns {found, display_name, lat, lon, type, "
                "category, importance, osm_id, osm_type, bounding_box, address}; importance below 0.1 marks an "
                "obscure or ambiguous match. DO NOT USE FOR: places shown only in an image, or travel times."),
            input_schema={"type": "object", "properties": {
                "query": {"type": "string", "description": "Place name: a city, region, country, landmark or address."},
                "country_code": {"type": "string", "description": "Optional ISO 3166-1 alpha-2 filter, e.g. 'jp'."}},
                "required": ["query"]})

    def execute(self, query="", country_code=""):
        if not query.strip():
            return {"found": False, "error": "query is required"}
        params = {"q": query.strip(), "format": "jsonv2", "addressdetails": 1, "limit": 1}
        if country_code.strip():
            params["countrycodes"] = country_code.strip().lower()
        try:
            response = requests.get(SEARCH, params=params, headers=HEADERS, timeout=20)
            response.raise_for_status()
            places = response.json()
        except (requests.RequestException, ValueError) as exc:
            return {"found": False, "error": f"Nominatim request failed: {exc}"}
        if not places:
            return {"found": False}
        place = places[0]
        return {"found": True, "display_name": place.get("display_name", ""),
                "lat": float(place["lat"]) if place.get("lat") else None,
                "lon": float(place["lon"]) if place.get("lon") else None,
                "type": place.get("type", ""), "category": place.get("category", ""),
                "importance": place.get("importance"), "osm_id": place.get("osm_id"),
                "osm_type": place.get("osm_type", ""),
                "bounding_box": [float(value) for value in place.get("boundingbox") or []],
                "address": place.get("address", {})}
