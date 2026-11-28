from openvlharness.toolsmith import GeneratedTool, ToolCallError


class BestLabelByImage(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="best_label_by_image",
            tool_description=("Choose which of several candidate text labels best describes an image, e.g. a season, a "
                              "material or an architectural style. DO NOT USE FOR: open-ended naming or reading text."),
            input_schema={"type": "object", "properties": {
                "image_id": {"type": "string", "description": "Image to classify."},
                "labels": {"type": "array", "items": {"type": "string"}, "description": "Two or more candidate labels."}},
                "required": ["image_id", "labels"]})

    def execute(self, image_base64=None, labels=None):
        labels = [label for label in labels or [] if isinstance(label, str) and label.strip()]
        if not image_base64 or len(labels) < 2:
            return {"error": "an image and at least two labels are required"}
        try:
            ranking = self.call_tool("image_similarity", image_base64=image_base64, candidate_labels=labels)
        except ToolCallError as exc:
            return {"error": str(exc)}
        if ranking.get("best_score", 0) < 0.3:
            return {"error": "no label fits the image confidently", "ranked": ranking.get("ranked", [])[:3]}
        return ranking["best"]
