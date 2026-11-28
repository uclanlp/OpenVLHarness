from openvlharness.toolsmith import GeneratedTool, ToolCallError


class EntityPropertyLookup(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="entity_property_lookup",
            tool_description=("Look up one property of a named entity in Wikidata, e.g. the architect of a building or "
                              "the inflow of a lake. DO NOT USE FOR: entities known only from an image, or opinions."),
            input_schema={"type": "object", "properties": {
                "entity": {"type": "string", "description": "Entity name or Wikidata QID."},
                "property": {"type": "string", "description": "Property to read, e.g. 'architect'."}},
                "required": ["entity", "property"]})

    def execute(self, entity="", property=""):
        if not (entity.strip() and property.strip()):
            return {"error": "entity and property are required"}
        try:
            record = self.call_tool("wikidata_query", entity=entity, property=property)
        except ToolCallError as exc:
            return {"error": str(exc)}
        matched = record.get("matched") or {}
        if not record.get("found") or matched.get("confidence", 0) < 1.0 or not matched.get("values"):
            return {"error": f"Wikidata has no confident '{property}' for {entity}"}
        return ", ".join(matched["values"][:3])
