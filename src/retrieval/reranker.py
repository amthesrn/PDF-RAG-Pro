"""Reranker using a cross-encoder (sentence-transformers) to rerank candidate chunks."""

from __future__ import annotations

from typing import List

from sentence_transformers import CrossEncoder

from config.settings import settings
from src.models.schemas import RetrievedChunk


class Reranker:
    """Reranks retrieved chunks using a cross-encoder."""

    def __init__(self, min_score: float = -2.0) -> None:
        self._model_name = settings.reranker_model
        self._model = CrossEncoder(self._model_name)
        self._min_score = min_score

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """Return top_k chunks after reranking by cross-encoder score."""
        if not candidates:
            return []

        # Prepare pairs: (query, chunk_text)
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)

        # Strict thresholding: drop chunks that the cross-encoder finds completely irrelevant
        filtered = [c for c in candidates if c.rerank_score >= self._min_score]

        sorted_chunks = sorted(filtered, key=lambda c: c.rerank_score, reverse=True)
        return sorted_chunks[:top_k]
