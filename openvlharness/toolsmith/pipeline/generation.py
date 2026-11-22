"""Phase 3: classify the expected answer shape, then ask the generator LLM to write a tool."""
import json
import re
from pathlib import Path

from .prompts import prompt

EXAMPLES = Path(__file__).with_name("examples")
BASE_RESULTS = {
    "Text_Search_Tool": "summarized results with source URLs as text, or a mapping whose \"text\" holds that text",
    "Image_Search_Tool": "visual matches with titles and source URLs as text, or a mapping whose \"text\" holds that text",
    "Webpage_Visit_Tool": "text with the evidence and a summary extracted for the goal",
    "OCR_Tool": "{\"text\": the transcription}",
    "Zoom_In_Tool": "{\"cropped_image_base64\": the crop as PNG base64}",
}


def describe_tools(base, catalog):
    """List the tools a generated tool may call, with their arguments and result shapes.

    Args:
        base: Mapping of base tool names to their public function schemas.
        catalog: Mapping of generated tool names to tool instances.
    """
    lines = [_entry(name, base[name]["description"], base[name]["parameters"], result)
             for name, result in BASE_RESULTS.items() if name in base]
    lines += [_entry(name, tool.tool_description, tool.input_schema, "the value its description names")
              for name, tool in catalog.items()]
    return "\n".join(lines)


def _entry(name, description, parameters, result):
    arguments = ", ".join(re.sub(r"image_id$", "image_base64", key) for key in parameters.get("properties", {}))
    return f"- {name}({arguments}): {description} Returns {result}."


def source_of(reply):
    """Return the Python source in a generator reply, or None when it holds none."""
    text = reply or ""
    found = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        code = json.loads(found.group(0)).get("code") if found else None
    except (ValueError, AttributeError):
        code = None
    if not isinstance(code, str):
        block = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        code = block.group(1) if block else None
    return code.strip() + "\n" if code and code.strip() else None


class Generator:
    """Write and revise candidate tools; the analyst model classifies answer shapes."""

    def __init__(self, model, analyst, tools):
        """Build the generator system prompt from the callable tools and the seed examples.

        Args:
            model: AuxiliaryModel that writes tools.
            analyst: AuxiliaryModel that classifies the expected answer shape.
            tools: Text listing the callable tools, from describe_tools.
        """
        self.model = model
        self.analyst = analyst
        examples = "\n\n".join(path.read_text() for path in sorted(EXAMPLES.glob("[!_]*.py")))
        self.system = prompt("generator.system").format(catalog=tools, examples=examples)

    def shape(self, question):
        """Return the classifier's answer type, answer length and extraction hint for a question."""
        reply = self.analyst.generate(prompt("classifier.user").format(question=question),
                                      system_prompt=prompt("classifier.system"), max_tokens=256)
        fields = re.findall(r"^(ANSWER_TYPE|ANSWER_LENGTH|EXTRACTION_HINT):\s*(.+)$", reply or "", re.MULTILINE)
        return {key.lower(): value.strip() for key, value in fields}

    def write(self, gap):
        """Return the source of a first candidate for a gap, or None."""
        shape = self.shape(gap.question)
        image = ("\nImage: provided with the query\n  (Declare image_id in input_schema and chain-call a vision tool)\n"
                 if gap.image else "")
        return self._source(prompt("generator.user").format(
            question=gap.question, answer_type=shape.get("answer_type", "OTHER"),
            answer_length=shape.get("answer_length", "short_phrase"), extraction_hint=shape.get("extraction_hint", ""),
            image_note=image, reason=gap.reason or "unknown",
            tried_tools=", ".join(sorted(set(gap.tried_tools))) or "none"))

    def revise(self, gap, code, feedback):
        """Return the source of a candidate revised after a gate rejection, or None."""
        return self._source(prompt("generator.feedback").format(question=gap.question, feedback=feedback, code=code))

    def _source(self, request):
        return source_of(self.model.generate(request, system_prompt=self.system, max_tokens=16384))
