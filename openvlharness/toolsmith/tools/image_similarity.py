import threading
from functools import cache

from openvlharness.services.tools.image_utils import load_image_base64
from openvlharness.toolsmith import GeneratedTool

LOCK = threading.Lock()


@cache
def clip():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return model.eval(), preprocess, open_clip.get_tokenizer("ViT-B-32")


class ImageSimilarity(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="image_similarity",
            tool_description=(
                "Rank candidate text labels by how well they describe an image, with CLIP zero-shot scoring. Returns "
                "{ranked: [{candidate, score}], best, best_score, mode}; scores are softmax probabilities, and "
                "best_score below 0.3 is a weak match. DO NOT USE FOR: reading text or naming unlisted entities."),
            input_schema={"type": "object", "properties": {
                "image_id": {"type": "string", "description": "Image to score."},
                "candidate_labels": {"type": "array", "items": {"type": "string"},
                                     "description": "Two or more candidate descriptions."}},
                "required": ["image_id", "candidate_labels"]})

    def execute(self, image_base64=None, candidate_labels=None):
        labels = [label.strip() for label in candidate_labels or [] if isinstance(label, str) and label.strip()]
        if not image_base64 or len(labels) < 2:
            return {"error": "an image and at least two candidate_labels are required"}
        import torch

        with LOCK:
            model, preprocess, tokenizer = clip()
        with torch.no_grad():
            image = model.encode_image(preprocess(load_image_base64(image_base64)).unsqueeze(0))
            text = model.encode_text(tokenizer([f"a photo of {label}." for label in labels]))
            image, text = image / image.norm(dim=-1, keepdim=True), text / text.norm(dim=-1, keepdim=True)
            scores = (model.logit_scale.exp() * image @ text.T).softmax(dim=-1)[0].tolist()
        ranked = sorted(({"candidate": label, "score": round(score, 6)} for label, score in zip(labels, scores)),
                        key=lambda entry: -entry["score"])
        return {"ranked": ranked, "best": ranked[0]["candidate"], "best_score": ranked[0]["score"], "mode": "text"}
