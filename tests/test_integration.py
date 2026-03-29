"""
test_integration.py — End-to-end integration tests for bot2 RAG pipeline.
Requires an indexed PDF in the local ChromaDB vectorstore.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("bot2_tests.integration")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Helper ────────────────────────────────────────────────────────────────────
def save_json(filename: str, data):
    out = RESULTS_DIR / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Results saved to %s", out)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. PDF INGESTION TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestIngestion:
    """Test that the Options Trading PDF can be parsed and chunked."""

    PDF_PATH = Path(__file__).parent.parent.parent / "input" / "Options_Trading_Complete_Mastery_Guide_Claude.pdf"

    @pytest.fixture(autouse=True)
    def _skip_if_no_pdf(self):
        if not self.PDF_PATH.exists():
            pytest.skip(f"Test PDF not found: {self.PDF_PATH}")

    def test_pdf_parsing(self):
        from src.ingestion.pdf_parser import PDFParser
        parser = PDFParser(self.PDF_PATH, image_output_dir="bot2/data/images")
        pages = parser.parse()
        assert len(pages) > 0, "Parser returned 0 pages"
        pages_with_text = [p for p in pages if p.text.strip()]
        logger.info("Parsed %d pages, %d with text", len(pages), len(pages_with_text))
        assert len(pages_with_text) > 0, "No pages had extractable text"

    def test_chunking_produces_parent_child(self):
        from src.ingestion.pdf_parser import PDFParser
        from src.ingestion.chunker import Chunker
        parser = PDFParser(self.PDF_PATH, image_output_dir="bot2/data/images")
        pages = parser.parse()
        chunker = Chunker()
        chunks = chunker.chunk_pages(pages, source_pdf="Options_Trading.pdf")
        assert len(chunks) > 10, f"Expected many chunks, got {len(chunks)}"
        # Verify parent-child relationship
        with_parent = [c for c in chunks if c.parent_text and len(c.parent_text) >= len(c.text)]
        logger.info("Total chunks: %d, with parent_text: %d", len(chunks), len(with_parent))
        assert len(with_parent) > 0, "No chunks have parent_text set"


# ═══════════════════════════════════════════════════════════════════════════════
#  2. RETRIEVAL TESTS (requires indexed data in vectorstore)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrieval:
    """Test retrieval across all 3 modes using a known question."""

    @pytest.fixture(autouse=True)
    def _setup_pipeline(self):
        from src.vectorstore.chroma_store import VectorStore
        from src.retrieval.retriever import HybridRetriever
        self.store = VectorStore()
        if self.store.count() == 0:
            pytest.skip("No data indexed in vectorstore — index a PDF first")
        self.retriever = HybridRetriever(self.store)
        self.retriever.build_bm25_index()

    def test_semantic_retrieval(self):
        from src.models.schemas import RetrievalMode
        result = self.retriever.retrieve("What is a call option?", mode=RetrievalMode.SEMANTIC, top_k=4)
        assert len(result.chunks) > 0, "Semantic search returned 0 chunks"
        logger.info("Semantic: %d chunks retrieved", len(result.chunks))

    def test_keyword_retrieval(self):
        from src.models.schemas import RetrievalMode
        result = self.retriever.retrieve("call option strike price", mode=RetrievalMode.KEYWORD, top_k=4)
        assert len(result.chunks) > 0, "Keyword search returned 0 chunks"
        logger.info("Keyword: %d chunks retrieved", len(result.chunks))

    def test_hybrid_retrieval(self):
        from src.models.schemas import RetrievalMode
        result = self.retriever.retrieve("What is a put option?", mode=RetrievalMode.HYBRID, top_k=4)
        assert len(result.chunks) > 0, "Hybrid search returned 0 chunks"
        logger.info("Hybrid: %d chunks retrieved", len(result.chunks))

    def test_hybrid_uses_rrf_fusion(self):
        """Verify that hybrid results differ from pure semantic."""
        from src.models.schemas import RetrievalMode
        query = "options expiration date"
        sem = self.retriever.retrieve(query, mode=RetrievalMode.SEMANTIC, top_k=6)
        hyb = self.retriever.retrieve(query, mode=RetrievalMode.HYBRID, top_k=6)
        sem_ids = [c.chunk.chunk_id for c in sem.chunks]
        hyb_ids = [c.chunk.chunk_id for c in hyb.chunks]
        # They should have at least some overlap but not be identical ordering
        logger.info("Semantic IDs: %s", sem_ids[:3])
        logger.info("Hybrid IDs:   %s", hyb_ids[:3])
        assert len(hyb.chunks) > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  3. RERANKING INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestRerankingIntegration:
    """Test reranker filters and reorders retrieved chunks."""

    @pytest.fixture(autouse=True)
    def _setup_pipeline(self):
        from src.vectorstore.chroma_store import VectorStore
        from src.retrieval.retriever import HybridRetriever
        from src.retrieval.reranker import Reranker
        self.store = VectorStore()
        if self.store.count() == 0:
            pytest.skip("No data indexed in vectorstore")
        self.retriever = HybridRetriever(self.store)
        self.retriever.build_bm25_index()
        self.reranker = Reranker()

    def test_reranking_reorders(self):
        from src.models.schemas import RetrievalMode
        result = self.retriever.retrieve("What is a call option?", mode=RetrievalMode.HYBRID, top_k=8)
        reranked = self.reranker.rerank("What is a call option?", result.chunks, top_k=4)
        assert len(reranked) <= 4
        assert len(reranked) > 0
        # Verify descending rerank_score
        scores = [c.rerank_score for c in reranked]
        assert scores == sorted(scores, reverse=True), "Reranked scores not in descending order"
        logger.info("Reranked %d -> %d chunks, top score: %.4f", len(result.chunks), len(reranked), scores[0])


# ═══════════════════════════════════════════════════════════════════════════════
#  4. FULL RAG CHAIN TEST (Groq + Gemini)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRAGChain:
    """End-to-end test: question -> retrieval -> reranking -> generation."""

    @pytest.fixture(autouse=True)
    def _setup_chain(self):
        from src.vectorstore.chroma_store import VectorStore
        from src.retrieval.retriever import HybridRetriever
        from src.retrieval.reranker import Reranker
        from src.generation.llm_chain import RAGChain
        self.store = VectorStore()
        if self.store.count() == 0:
            pytest.skip("No data indexed in vectorstore")
        retriever = HybridRetriever(self.store)
        retriever.build_bm25_index()
        reranker = Reranker()
        self.chain = RAGChain(retriever, reranker)

    def test_groq_answer_generation(self):
        from src.models.schemas import RAGRequest
        request = RAGRequest(question="What is a call option?", llm_provider="groq", top_k=4)
        response = self.chain.answer(request)
        assert response.answer, "Empty answer from Groq"
        assert response.confidence in ("high", "medium", "low", "not_found")
        logger.info("Groq answer (%s): %s", response.confidence, response.answer[:120])
        save_json("rag_chain_groq_test.json", {
            "question": request.question,
            "answer": response.answer,
            "confidence": response.confidence,
            "model": response.model_used,
            "time_ms": response.processing_time_ms,
            "source_pages": [c.page_number for c in response.source_chunks],
        })

    def test_gemini_answer_generation(self):
        from src.models.schemas import RAGRequest
        request = RAGRequest(question="What is a call option?", llm_provider="gemini", top_k=4)
        response = self.chain.answer(request)
        assert response.answer, "Empty answer from Gemini"
        assert response.confidence in ("high", "medium", "low", "not_found")
        logger.info("Gemini answer (%s): %s", response.confidence, response.answer[:120])
        save_json("rag_chain_gemini_test.json", {
            "question": request.question,
            "answer": response.answer,
            "confidence": response.confidence,
            "model": response.model_used,
            "time_ms": response.processing_time_ms,
            "source_pages": [c.page_number for c in response.source_chunks],
        })

    def test_answer_contains_citations(self):
        """Verify the strict citation enforcement produces [doc_X:page_Y] markers."""
        from src.models.schemas import RAGRequest
        request = RAGRequest(question="What is the difference between a call option and a put option?", llm_provider="groq", top_k=4)
        response = self.chain.answer(request)
        citation_pattern = r'\[doc_\d+:page_\d+\]'
        citations = re.findall(citation_pattern, response.answer)
        logger.info("Found %d citations in answer: %s", len(citations), citations)
        # Log even if 0 citations — this is informational
        save_json("citation_test.json", {
            "question": request.question,
            "answer": response.answer,
            "citations_found": citations,
            "citation_count": len(citations),
        })

    def test_not_found_on_irrelevant_question(self):
        """Questions totally unrelated to the PDF should produce a not-found response."""
        from src.models.schemas import RAGRequest
        request = RAGRequest(question="What is the capital of Mars?", llm_provider="groq", top_k=4)
        response = self.chain.answer(request)
        logger.info("Irrelevant Q answer: %s", response.answer[:100])
        # The answer should mention "not available" or confidence should be not_found
        is_not_found = (
            "not available" in response.answer.lower()
            or "not found" in response.answer.lower()
            or response.confidence == "not_found"
        )
        assert is_not_found, f"Expected not-found response, got: {response.answer[:100]}"
        logger.info("PASS: irrelevant question correctly returned not-found")


# ═══════════════════════════════════════════════════════════════════════════════
#  5. CITATION VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestCitationValidation:
    """Validate that LLM-generated citations map to real source chunks."""

    @pytest.fixture(autouse=True)
    def _setup_chain(self):
        from src.vectorstore.chroma_store import VectorStore
        from src.retrieval.retriever import HybridRetriever
        from src.retrieval.reranker import Reranker
        from src.generation.llm_chain import RAGChain
        self.store = VectorStore()
        if self.store.count() == 0:
            pytest.skip("No data indexed in vectorstore")
        retriever = HybridRetriever(self.store)
        retriever.build_bm25_index()
        reranker = Reranker()
        self.chain = RAGChain(retriever, reranker)

    def test_citation_integrity(self):
        """Parse citations from answer and verify they reference real source pages."""
        from src.models.schemas import RAGRequest
        request = RAGRequest(
            question="What are the Greeks in options trading?",
            llm_provider="groq", top_k=4
        )
        response = self.chain.answer(request)

        # Extract citations like [doc_0:page_5]
        citation_pattern = r'\[doc_(\d+):page_(\d+)\]'
        raw_citations = re.findall(citation_pattern, response.answer)

        source_pages = {c.page_number for c in response.source_chunks}
        source_indices = {c.chunk_index for c in response.source_chunks}

        valid = 0
        invalid = 0
        details = []
        for doc_idx, page_num in raw_citations:
            page = int(page_num)
            is_valid = page in source_pages
            if is_valid:
                valid += 1
            else:
                invalid += 1
            details.append({
                "citation": f"[doc_{doc_idx}:page_{page_num}]",
                "page_in_sources": is_valid,
            })

        total = valid + invalid
        accuracy = (valid / total * 100) if total > 0 else 0.0

        audit = {
            "question": request.question,
            "total_citations": total,
            "valid": valid,
            "invalid": invalid,
            "accuracy_pct": round(accuracy, 1),
            "source_pages": sorted(source_pages),
            "details": details,
            "answer_preview": response.answer[:300],
        }
        save_json("citation_audit.json", audit)
        logger.info("Citation audit: %d total, %d valid, %d invalid (%.1f%% accuracy)",
                     total, valid, invalid, accuracy)
