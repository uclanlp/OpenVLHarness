"""Per-query retrieval of generated tools by sentence-encoder similarity."""
import threading

import numpy as np

ENCODER = "BAAI/bge-m3"
TOP_K = 10


class ToolRetriever:
    """Rank generated tools by the cosine similarity of query and description embeddings."""

    def __init__(self, descriptions, *, encoder=ENCODER, top_k=TOP_K):
        """Load the sentence encoder and embed every tool description once.

        Args:
            descriptions: Mapping of tool names to tool descriptions.
            encoder: Sentence-transformers model name or local path.
            top_k: Number of tools returned per query.
        """
        from sentence_transformers import SentenceTransformer

        self.names = list(descriptions)
        self.top_k = top_k
        self.model = SentenceTransformer(encoder)
        self.lock = threading.Lock()
        self.embeddings = self.encode([descriptions[name] for name in self.names])

    def encode(self, texts):
        """Return unit-normalized embeddings for a list of texts."""
        with self.lock:
            return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def retrieve(self, query):
        """Return the names of the top-K tools for a query, most similar first."""
        if not self.names:
            return []
        scores = self.embeddings @ self.encode([query])[0]
        return [self.names[index] for index in np.argsort(-scores, kind="stable")[:self.top_k]]
