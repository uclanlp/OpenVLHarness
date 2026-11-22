"""TF-IDF similarity between a text and tool descriptions, for Phase 2 and the catalog update."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def closest(text, descriptions):
    """Return the name and cosine similarity of the description most similar to text.

    Args:
        text: Gap text (query and failure reason) or a candidate tool's description.
        descriptions: Mapping of tool names to descriptions.

    Returns:
        (name, similarity), or (None, 0.0) when nothing is comparable.
    """
    names = [name for name, description in descriptions.items() if description.strip()]
    if not names or not text.strip():
        return None, 0.0
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit_transform(
            [descriptions[name] for name in names] + [text])
    except ValueError:
        return None, 0.0
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    best = int(scores.argmax())
    return names[best], float(scores[best])
