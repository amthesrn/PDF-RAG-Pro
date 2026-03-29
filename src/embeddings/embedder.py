"""Embedding utilities using local HuggingFace models (BGE-Large by default)."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings


class Embedder:
    """Wrapper around sentence-transformers for generating embeddings."""

    def __init__(self) -> None:
        self._model_name = settings.embedding_model
        self._model = SentenceTransformer(self._model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents and return normalized vectors."""
        if not texts:
            return []
        embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = self._normalize(embeddings)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query and return normalized vector."""
        emb = self._model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        emb = self._normalize(emb)
        return emb[0].tolist()

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
