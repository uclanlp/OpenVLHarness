import re

from openvlharness.toolsmith import GeneratedTool, ToolCallError


class FossilIdentification(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="fossil_identification",
            tool_description=("Identify the species of a fossil specimen from the label or caption printed next to it in "
                              "a museum photo. DO NOT USE FOR: living animals, unlabeled specimens, or rock types."),
            input_schema={"type": "object", "properties": {
                "image_id": {"type": "string", "description": "Image showing the specimen and its label."}},
                "required": ["image_id"]})

    def execute(self, image_base64=None):
        if not image_base64:
            return {"error": "an image is required"}
        try:
            label = self.call_tool("OCR_Tool", image_base64=image_base64).get("text", "")
        except ToolCallError as exc:
            return {"error": str(exc)}
        binomial = re.search(r"\b([A-Z][a-z]+ [a-z]{3,})\b", label)
        if not binomial:
            return {"error": "no scientific name on the label", "label": label[:200]}
        name = binomial.group(1)
        try:
            found = self.call_tool("Text_Search_Tool", query=f"{name} fossil species")
        except ToolCallError as exc:
            return {"error": str(exc), "candidate": name}
        text = found.get("text", "") if isinstance(found, dict) else found
        return name if name.lower() in text.lower() else {"error": f"{name} is not confirmed by search", "candidate": name}
