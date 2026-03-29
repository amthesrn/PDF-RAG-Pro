"""ChromaDB vector store wrapper.

Provides a simple interface for adding chunks, querying semantic similarity,
and managing the collection.
"""

from __future__ import annotations

from typing import List, Optional

import chromadb

from config.settings import settings
from src.models.schemas import DocumentChunk, RetrievedChunk


class VectorStoreError(Exception):
    pass


class VectorStore:
    def __init__(self) -> None:
        # Bypassing the Chroma Cloud 300-chunk free tier limit by reverting to local storage.
        # Parent-Child chunking inherently generates more chunks for higher precision.
        self._client = chromadb.PersistentClient(path=str(settings.vectorstore_dir))
        self._col = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"source": "pdf_rag_chatbot"},
        )

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return
            
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunks and embeddings must have the same length")

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]

        try:
            self._col.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to add chunks to ChromaDB: {exc}") from exc

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        """Return top_k chunks ordered by semantic similarity."""
        if not query.strip():
            return []

        # Use the same embedding model as the rest of the pipeline.
        from src.embeddings.embedder import Embedder

        embedder = Embedder()
        query_emb = embedder.embed_query(query)

        results = self._col.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        chunks: List[RetrievedChunk] = []
        for doc, meta, dist in zip(docs, metas, dists):
            if not meta:
                continue
            try:
                chunk = DocumentChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc,
                    page_number=int(meta.get("page_number", 0)),
                    content_type=meta.get("content_type", "text"),
                    section=meta.get("section", ""),
                    section_heading=meta.get("section_heading", ""),
                    source_pdf=meta.get("source_pdf", ""),
                )
            except Exception:
                continue

            # Chroma returns distances (lower is better). When using cosine,
            # distance is 1 - cosine_similarity.
            semantic_score = 1.0 - float(dist) if isinstance(dist, (int, float)) else 0.0
            semantic_score = max(0.0, min(1.0, semantic_score))

            chunks.append(RetrievedChunk(chunk=chunk.model_dump(), semantic_score=semantic_score))

        return chunks

    def count(self) -> int:
        return self._col.count()

    def delete_collection(self) -> None:
        self._client.delete_collection(name=settings.collection_name)
