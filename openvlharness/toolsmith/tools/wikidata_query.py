import re
import time

import requests

from openvlharness.toolsmith import GeneratedTool

HEADERS = {"User-Agent": "OpenVLHarness-ToolSmith/0.1 (https://github.com/uclanlp/OpenVLHarness)",
           "Accept": "application/json"}
API = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"
CLAIMS = ("SELECT ?propertyLabel ?valueLabel WHERE {{ wd:{qid} ?direct ?value . "
          "?property wikibase:directClaim ?direct . "
          'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}')
MAX_PROPERTIES = 250


def get_json(url, params):
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=25)
            if response.status_code == 200:
                return response.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url} did not answer")


def words(text):
    return {word for word in re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split() if len(word) > 2}


class WikidataQuery(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="wikidata_query",
            tool_description=(
                "Structured lookup of a real-world entity in Wikidata. Resolves an entity name or QID and returns "
                "{found, qid, entity_label, entity_description, claims: {property label: [values]}}; with a "
                "property hint it adds matched: {property, values, confidence}, where confidence >= 1.0 means the "
                "hint matched a property directly. DO NOT USE FOR: image recognition, OCR, or fictional entities."),
            input_schema={"type": "object", "properties": {
                "entity": {"type": "string", "description": "Entity name, or a QID such as Q243."},
                "property": {"type": "string", "description": "Optional property hint, e.g. 'architect'."}},
                "required": ["entity"]})

    def execute(self, entity="", property=""):
        entity = entity.strip()
        if not entity:
            return {"found": False, "error": "entity is required"}
        best = None
        try:
            for qid, label, description in self.candidates(entity)[:3]:
                claims = self.claims(qid)
                result = {"found": True, "qid": qid, "entity_label": label, "entity_description": description,
                          "n_properties": len(claims), "claims": claims}
                if not property.strip():
                    return result
                result["matched"] = self.match(claims, property)
                if result["matched"] and result["matched"]["confidence"] >= 1.0:
                    return result
                best = best or result
        except RuntimeError as exc:
            return {"found": False, "error": str(exc)}
        return best or {"found": False, "error": f"no Wikidata entity matches {entity!r}"}

    @staticmethod
    def candidates(entity):
        """Return (qid, label, description) for a QID, or for search hits with exact label matches first."""
        if re.fullmatch(r"Q\d+", entity):
            record = get_json(API, {"action": "wbgetentities", "ids": entity, "props": "labels|descriptions",
                                    "languages": "en", "format": "json"}).get("entities", {}).get(entity, {})
            if "missing" in record:  # Wikidata has no item with this QID
                return []
            return [(entity, record.get("labels", {}).get("en", {}).get("value", entity),
                     record.get("descriptions", {}).get("en", {}).get("value", ""))]
        hits = get_json(API, {"action": "wbsearchentities", "search": entity, "language": "en", "uselang": "en",
                              "format": "json", "type": "item", "limit": 5}).get("search") or []
        hits.sort(key=lambda hit: hit.get("label", "").lower() != entity.lower())
        return [(hit["id"], hit.get("label", entity), hit.get("description", "")) for hit in hits]

    @staticmethod
    def claims(qid):
        claims = {}
        for row in get_json(SPARQL, {"query": CLAIMS.format(qid=qid), "format": "json"})["results"]["bindings"]:
            name, value = row.get("propertyLabel", {}).get("value", ""), row.get("valueLabel", {}).get("value", "")
            if not name or not value or value in claims.get(name, []):
                continue
            if name in claims or len(claims) < MAX_PROPERTIES:
                claims.setdefault(name, []).append(value)
        return claims

    @staticmethod
    def match(claims, hint):
        wanted = words(hint)

        def overlap(name):
            # An exact token counts 1 and a token found only as a substring 0.5, so confidence
            # reaches 1.0 only when the property name contains every hint token.
            have = words(name)
            return len(wanted & have) + 0.5 * sum(any(w in h or h in w for h in have) for w in wanted - have)
        best = max(claims, key=overlap, default=None)
        if not wanted or best is None or overlap(best) == 0:
            return None
        return {"property": best, "values": claims[best], "score": overlap(best), "hint_token_count": len(wanted),
                "confidence": round(overlap(best) / len(wanted), 3)}
