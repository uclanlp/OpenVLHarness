import re

from openvlharness.toolsmith import GeneratedTool, ToolCallError


def page_answers(page):
    """Yield the first summary line of each fetched page; failed fetches carry no answer.

    Each page's block starts with "The useful information in" and ends with its summary,
    after the page's own text, which may hold "=======" or "Summary:" lines too.
    """
    for block in str(page or "").split("\n\n=======\n\nThe useful information in "):
        summary = block.rsplit("\n\nSummary: \n", 1)[1].strip() if "\n\nSummary: \n" in block else ""
        if summary and not summary.startswith(("FETCH FAILED", "No summary generated")):
            yield summary.splitlines()[0][:120]


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
        return next(page_answers(page), None) or {"error": "the cast pages do not name the performer"}
