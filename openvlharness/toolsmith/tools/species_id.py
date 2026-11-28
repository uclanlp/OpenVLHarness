import threading
from functools import cache

from openvlharness.services.tools.image_utils import load_image_base64
from openvlharness.toolsmith import GeneratedTool

LOCK = threading.Lock()


@cache
def bioclip():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    return model.eval(), preprocess, open_clip.get_tokenizer("hf-hub:imageomics/bioclip")


class SpeciesId(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="species_id",
            tool_description=(
                "Score candidate species for an organism photo with BioCLIP zero-shot classification. Returns "
                "{ranked: [{name, score}], best, best_score}; scores are softmax probabilities over the candidates, "
                "and best_score below 0.3 is a weak match. DO NOT USE FOR: people, objects, or unlisted species."),
            input_schema={"type": "object", "properties": {
                "image_id": {"type": "string", "description": "Photo of the organism."},
                "candidate_species": {"type": "array", "items": {"type": "string"},
                                      "description": "Two or more species names, scientific or common."},
                "top_k": {"type": "integer", "description": "Number of ranked candidates to return (default 5)."}},
                "required": ["image_id", "candidate_species"]})

    def execute(self, image_base64=None, candidate_species=None, top_k=5):
        species = [name.strip() for name in candidate_species or [] if isinstance(name, str) and name.strip()]
        if not image_base64 or len(species) < 2:
            return {"error": "an image and at least two candidate_species are required"}
        import torch

        with LOCK:
            model, preprocess, tokenizer = bioclip()
        with torch.no_grad():
            image = model.encode_image(preprocess(load_image_base64(image_base64)).unsqueeze(0))
            text = model.encode_text(tokenizer([f"a photo of {name}." for name in species]))
            image, text = image / image.norm(dim=-1, keepdim=True), text / text.norm(dim=-1, keepdim=True)
            scores = (model.logit_scale.exp() * image @ text.T).softmax(dim=-1)[0].tolist()
        ranked = sorted(({"name": name, "score": round(score, 6)} for name, score in zip(species, scores)),
                        key=lambda entry: -entry["score"])[:max(1, int(top_k))]
        return {"ranked": ranked, "best": ranked[0]["name"], "best_score": ranked[0]["score"]}
