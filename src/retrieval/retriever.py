"""Hybrid retrieval: semantic + keyword (BM25) with RRF fusion."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from config.settings import settings
from src.models.schemas import (
    DocumentChunk,
    RetrievalMode,
    RetrievalResult,
    RetrievedChunk,
)
from src.utils.logger import get_logger
from src.vectorstore.chroma_store import VectorStore

logger = get_logger(__name__)


class RetrieverError(Exception):
    """Raised when retrieval fails."""


class HybridRetriever:
    """Retrieves relevant chunks using hybrid (semantic + keyword) search."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._vs: VectorStore = vector_store
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[DocumentChunk] = []
        self._bm25_built = False

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int | None = None,
    ) -> RetrievalResult:
        if not query.strip():
            raise RetrieverError("Query cannot be empty")

        top_k = top_k or settings.top_k_retrieve
        logger.debug("Retrieving — mode: %s | query: %s", mode.value, query[:80])

        try:
            if mode == RetrievalMode.SEMANTIC:
                chunks = self._semantic_only(query, top_k)
            elif mode == RetrievalMode.KEYWORD:
                chunks = self._keyword_only(query, top_k)
            else:
                chunks = self._hybrid(query, top_k)
        except RetrieverError:
            raise
        except Exception as exc:
            logger.error("Retrieval error: %s", exc, exc_info=True)
            raise RetrieverError(f"Retrieval failed: {exc}") from exc

        return RetrievalResult(query=query, mode=mode, chunks=chunks)

    def build_bm25_index(self, all_chunks: list[DocumentChunk] | None = None) -> None:
        if all_chunks:
            docs = all_chunks
        else:
            docs = self._fetch_all_from_chroma()

        if not docs:
            logger.warning("No documents found for BM25 index")
            return

        tokenized = [self._tokenize(d.text) for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_docs = docs
        self._bm25_built = True
        logger.info("BM25 index built — %d documents", len(docs))

    def _semantic_only(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._vs.search(query, top_k=top_k)

    def _keyword_only(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self._ensure_bm25()
        return self._bm25_search(query, top_k)

    def _hybrid(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self._ensure_bm25()

        sem_results = self._vs.search(query, top_k=top_k)
        bm25_results = self._bm25_search(query, top_k=top_k)

        sem_ranks = {r.chunk.chunk_id: i + 1 for i, r in enumerate(sem_results)}
        kw_ranks = {r.chunk.chunk_id: i + 1 for i, r in enumerate(bm25_results)}

        all_ids = set(sem_ranks) | set(kw_ranks)
        rrf_k = 60

        chunk_map: dict[str, RetrievedChunk] = {}
        for r in sem_results + bm25_results:
            chunk_map[r.chunk.chunk_id] = r

        scored: list[tuple[str, float]] = []
        for cid in all_ids:
            sem_rank = sem_ranks.get(cid, len(sem_results) + 1)
            kw_rank = kw_ranks.get(cid, len(bm25_results) + 1)
            rrf = (1 / (rrf_k + sem_rank)) + (1 / (rrf_k + kw_rank))
            scored.append((cid, rrf))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [cid for cid, _ in scored[:top_k]]

        results: list[RetrievedChunk] = []
        for cid in top_ids:
            r = chunk_map[cid]
            r.keyword_score = 1 / (rrf_k + kw_ranks.get(cid, top_k + 1))
            r.combined_score = dict(scored)[cid]
            results.append(r)

        logger.debug(
            "Hybrid RRF merged %d semantic + %d keyword → %d unique",
            len(sem_results),
            len(bm25_results),
            len(results),
        )
        return results

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)  # type: ignore[union-attr]

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[RetrievedChunk] = []
        for idx, score in indexed:
            if score <= 0:
                break
            chunk = self._bm25_docs[idx]
            results.append(
                RetrievedChunk(
                    chunk=chunk.model_dump(),
                    keyword_score=round(float(score), 4),
                )
            )
        return results

    def _ensure_bm25(self) -> None:
        if not self._bm25_built:
            logger.info("BM25 index not built yet — building lazily…")
            self.build_bm25_index()

    def _fetch_all_from_chroma(self) -> list[DocumentChunk]:
        try:
            col = self._vs._col
            total = col.count()
            if total == 0:
                return []
            raw = col.get(
                limit=total,
                include=["documents", "metadatas"],
            )
            chunks: list[DocumentChunk] = []
            for doc, meta in zip(raw["documents"], raw["metadatas"]):
                from src.models.schemas import ContentType

                chunks.append(
                    DocumentChunk(
                        chunk_id=meta.get("chunk_id", ""),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        text=doc,
                        page_number=int(meta.get("page_number", 0)),
                        content_type=ContentType(meta.get("content_type", "text")),
                        section=meta.get("section", ""),
                        section_heading=meta.get("section_heading", ""),
                        source_pdf=meta.get("source_pdf", ""),
                    )
                )
            return chunks
        except Exception as exc:
            logger.error("Failed to fetch all docs from Chroma: %s", exc, exc_info=True)
            return []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple lowercase word tokeniser."""
        return text.lower().split()
