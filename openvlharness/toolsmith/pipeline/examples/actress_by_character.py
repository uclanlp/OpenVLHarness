import re

from openvlharness.toolsmith import GeneratedTool, ToolCallError


class ActressByCharacter(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="actress_by_character",
            tool_description=("Find the actor or actress who plays a named character in a film or TV series. "
                              "DO NOT USE FOR: identifying people in images, biographies, or crew roles."),
            input_schema={"type": "object", "properties": {
                "character": {"type": "string", "description": "Character name, e.g. 'Hermione Granger'."},
                "work": {"type": "string", "description": "Optional film or series title."}},
                "required": ["character"]})

    def execute(self, character="", work=""):
        if not character.strip():
            return {"error": "character is required"}
        try:
            found = self.call_tool("Text_Search_Tool", query=f"{character} {work} cast played by".strip())
            text = found.get("text", "") if isinstance(found, dict) else found
            urls = re.findall(r"https?://\S+", text)
            if not urls:
                return {"error": f"no cast page found for {character}"}
            page = self.call_tool("Webpage_Visit_Tool", urls=urls[:2],
                                  goal=f"Name the actor or actress who plays {character} {work}. Reply with the name only.")
        except ToolCallError as exc:
            return {"error": str(exc)}
        summary = page.split("Summary:", 1)[-1].strip() if isinstance(page, str) else ""
        return summary.splitlines()[0][:120] if summary else {"error": "the cast page does not name the performer"}
