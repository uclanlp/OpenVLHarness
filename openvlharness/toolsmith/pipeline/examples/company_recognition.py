import re

from openvlharness.toolsmith import GeneratedTool, ToolCallError


class CompanyRecognition(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="company_recognition",
            tool_description=("Identify the company or brand whose logo appears in an image: reverse image search proposes "
                              "a name, and a web search confirms it is a company. DO NOT USE FOR: people, places, or artwork."),
            input_schema={"type": "object", "properties": {
                "image_id": {"type": "string", "description": "Image containing the logo."}},
                "required": ["image_id"]})

    def execute(self, image_base64=None):
        if not image_base64:
            return {"error": "an image is required"}
        try:
            matches = self.call_tool("Image_Search_Tool", image_base64=image_base64)
        except ToolCallError as exc:
            return {"error": str(exc)}
        text = matches.get("text", "") if isinstance(matches, dict) else matches
        titles = re.findall(r"^\s*\d+\.\s+(.+)$", text, re.MULTILINE)
        candidate = re.split(r"\s[|\-–—:]\s", titles[0])[0].strip() if titles else ""
        if not candidate:
            return {"error": "no visual match names a company"}
        try:
            found = self.call_tool("Text_Search_Tool", query=f"{candidate} company official site")
        except ToolCallError as exc:
            return {"error": str(exc), "candidate": candidate}
        verified = found.get("text", "") if isinstance(found, dict) else found
        return candidate if candidate.lower() in verified.lower() else {"error": "the candidate is not confirmed", "candidate": candidate}
