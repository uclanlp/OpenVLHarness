"""Per-query retrieval of generated tools by sentence-encoder similarity."""
import re
import threading

import numpy as np

from .errors import EncoderError, describe

ENCODER = "BAAI/bge-m3"
TOP_K = 10


def _torch_version():
    """The installed PyTorch's (major, minor) version, or None without PyTorch."""
    try:
        import torch
    except Exception:  # not installed, or broken
        return None
    found = re.match(r"(\d+)\.(\d+)", getattr(torch, "__version__", ""))
    return (int(found.group(1)), int(found.group(2))) if found else None


def load_encoder(name):
    """Load a sentence-transformers model, on the first GPU when one is visible.

    Raises:
        EncoderError: sentence-transformers is not installed (the toolsmith extra), or the
            model cannot be loaded. The message quotes the original error, which is also
            its cause; it blames PyTorch only for transformers' refusal to read
            pytorch_model.bin weights (as BAAI/bge-m3's are) with PyTorch older than 2.6.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EncoderError(f"The sentence encoder {name!r} needs sentence-transformers, which the toolsmith extra "
                           f"installs: pip install -e '.[toolsmith]' ({describe(exc)})") from exc
    try:
        return SentenceTransformer(name)
    except Exception as exc:
        torch = _torch_version()
        if torch is not None and torch < (2, 6) and any("CVE-2025-32434" in describe(error) for error in _chain(exc)):
            raise EncoderError(
                f"The sentence encoder {name!r} has pytorch_model.bin weights, which transformers loads only with "
                f"PyTorch 2.6 or newer, and this is PyTorch {torch[0]}.{torch[1]}. Upgrade torch (pip install "
                "'torch>=2.6'), or set the encoder (toolsmith.encoder, or encoder in the pipeline configuration) to "
                f"a model with safetensors weights. ({describe(exc)})") from exc
        raise EncoderError(f"Cannot load the sentence encoder {name!r}: {describe(exc)}") from exc


def _chain(exc):
    """The exception and the ones it was raised from, a few deep."""
    found = []
    while exc is not None and len(found) < 5:
        found.append(exc)
        exc = exc.__cause__ or exc.__context__
    return found


class ToolRetriever:
    """Rank generated tools by the cosine similarity of query and description embeddings."""

    def __init__(self, descriptions, *, encoder=ENCODER, top_k=TOP_K):
        """Load the sentence encoder and embed every tool description once.

        Args:
            descriptions: Mapping of tool names to tool descriptions.
            encoder: Sentence-transformers model name or local path.
            top_k: Number of tools returned per query.

        Raises:
            EncoderError: The encoder cannot be loaded.
        """
        self.names = list(descriptions)
        self.top_k = top_k
        self.model = load_encoder(encoder)
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
