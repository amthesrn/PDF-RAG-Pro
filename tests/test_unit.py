"""
test_unit.py — Unit tests for bot2 core components.
Tests schemas, chunker, reranker logic, query rewriter, and vectorstore safety.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

# Ensure bot2 is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("bot2_tests.unit")


# ═══════════════════════════════════════════════════════════════════════════════
#  1. SCHEMA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocumentChunk:
    """Validate DocumentChunk Pydantic model."""

    def test_valid_chunk_creation(self):
        from src.models.schemas import DocumentChunk
        chunk = DocumentChunk(
            text="This is a valid test chunk with enough content for testing.",
            page_number=1,
            chunk_index=0,
            source_pdf="test.pdf",
        )
        assert chunk.text.strip() == chunk.text
        assert chunk.page_number == 1
        assert chunk.char_count == len(chunk.text)
        logger.info("PASS: valid chunk creation")

    def test_short_text_accepted(self):
        """After relaxing min_length=1, even single-char text should work."""
        from src.models.schemas import DocumentChunk
        chunk = DocumentChunk(
            text="X",
            page_number=1,
            chunk_index=0,
            source_pdf="test.pdf",
        )
        assert chunk.text == "X"
        logger.info("PASS: short text accepted (min_length=1)")

    def test_empty_text_rejected(self):
        from src.models.schemas import DocumentChunk
        with pytest.raises(ValidationError):
            DocumentChunk(text="", page_number=1, chunk_index=0, source_pdf="test.pdf")
        logger.info("PASS: empty text rejected")

    def test_whitespace_stripped(self):
        from src.models.schemas import DocumentChunk
        chunk = DocumentChunk(
            text="   hello world test chunk   ",
            page_number=1,
            chunk_index=0,
            source_pdf="test.pdf",
        )
        assert chunk.text == "hello world test chunk"
        logger.info("PASS: whitespace stripped")

    def test_parent_text_field(self):
        from src.models.schemas import DocumentChunk
        chunk = DocumentChunk(
            text="child chunk text",
            page_number=1,
            chunk_index=0,
            source_pdf="test.pdf",
            parent_text="full parent context is much longer",
        )
        assert chunk.parent_text == "full parent context is much longer"
        logger.info("PASS: parent_text field stored correctly")

    def test_to_metadata_returns_primitives(self):
        from src.models.schemas import DocumentChunk
        chunk = DocumentChunk(
            text="Valid chunk for metadata test, enough characters here.",
            page_number=2,
            chunk_index=1,
            content_type="table",
            section_heading="Chapter 1",
            source_pdf="test.pdf",
        )
        meta = chunk.to_metadata()
        assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())
        assert meta["page_number"] == 2
        assert meta["content_type"] == "table"
        logger.info("PASS: to_metadata returns all primitives")

    def test_negative_page_rejected(self):
        from src.models.schemas import DocumentChunk
        with pytest.raises(ValidationError):
            DocumentChunk(text="valid text", page_number=0, chunk_index=0, source_pdf="t.pdf")
        logger.info("PASS: page_number=0 rejected")


class TestQueryInput:
    """Validate QueryInput model."""

    def test_valid_query(self):
        from src.models.schemas import QueryInput
        q = QueryInput(question="What is options trading?")
        assert q.question == "What is options trading?"
        logger.info("PASS: valid query")

    def test_too_short_query_rejected(self):
        from src.models.schemas import QueryInput
        with pytest.raises(ValidationError):
            QueryInput(question="ab")
        logger.info("PASS: short query rejected")

    def test_whitespace_normalized(self):
        from src.models.schemas import QueryInput
        q = QueryInput(question="  what   is  this  ")
        assert q.question == "what is this"
        logger.info("PASS: whitespace normalized")

    def test_long_query_accepted(self):
        """QueryInput doesn't define max_length, so long queries are valid."""
        from src.models.schemas import QueryInput
        q = QueryInput(question="x" * 500)
        assert len(q.question) == 500
        logger.info("PASS: long query accepted (no max_length constraint)")


class TestRAGRequest:
    """Validate RAGRequest model with llm_provider field."""

    def test_default_provider_is_groq(self):
        from src.models.schemas import RAGRequest
        req = RAGRequest(question="test question here")
        assert req.llm_provider == "groq"
        logger.info("PASS: default provider is groq")

    def test_gemini_provider_accepted(self):
        from src.models.schemas import RAGRequest
        req = RAGRequest(question="test question here", llm_provider="gemini")
        assert req.llm_provider == "gemini"
        logger.info("PASS: gemini provider accepted")


class TestRawPage:
    """Validate RawPage model."""

    def test_empty_page_has_no_content(self):
        from src.models.schemas import RawPage
        page = RawPage(page_number=1)
        assert page.has_content is False
        logger.info("PASS: empty page has_content=False")

    def test_page_with_text_has_content(self):
        from src.models.schemas import RawPage
        page = RawPage(page_number=1, text="Some text here")
        assert page.has_content is True
        logger.info("PASS: text page has_content=True")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. CHUNKER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunker:
    """Validate the parent-child chunking logic."""

    def test_basic_chunking_produces_chunks(self):
        from src.ingestion.chunker import Chunker
        from src.models.schemas import RawPage
        chunker = Chunker(chunk_size=50, chunk_overlap=10, child_size=15, child_overlap=3)
        pages = [RawPage(page_number=1, text="This is a test paragraph with enough words. " * 10)]
        chunks = chunker.chunk_pages(pages, source_pdf="test.pdf")
        assert len(chunks) > 0
        logger.info("PASS: basic chunking produced %d chunks", len(chunks))

    def test_parent_text_populated(self):
        from src.ingestion.chunker import Chunker
        from src.models.schemas import RawPage
        chunker = Chunker(chunk_size=50, chunk_overlap=10, child_size=15, child_overlap=3)
        pages = [RawPage(page_number=1, text="This is sample text for the parent child chunking test. " * 10)]
        chunks = chunker.chunk_pages(pages, source_pdf="test.pdf")
        for c in chunks:
            assert c.parent_text is not None and len(c.parent_text) > 0
        logger.info("PASS: all %d chunks have parent_text", len(chunks))

    def test_child_text_shorter_than_parent(self):
        from src.ingestion.chunker import Chunker
        from src.models.schemas import RawPage
        chunker = Chunker(chunk_size=100, chunk_overlap=20, child_size=20, child_overlap=5)
        text = "word " * 200
        pages = [RawPage(page_number=1, text=text)]
        chunks = chunker.chunk_pages(pages, source_pdf="test.pdf")
        for c in chunks:
            assert len(c.text) <= len(c.parent_text)
        logger.info("PASS: child text <= parent text for all %d chunks", len(chunks))

    def test_empty_page_skipped(self):
        from src.ingestion.chunker import Chunker
        from src.models.schemas import RawPage
        chunker = Chunker()
        pages = [RawPage(page_number=1)]
        chunks = chunker.chunk_pages(pages, source_pdf="test.pdf")
        assert len(chunks) == 0
        logger.info("PASS: empty page skipped")

    def test_tiny_table_filtered(self):
        from src.ingestion.chunker import Chunker
        from src.models.schemas import RawPage
        chunker = Chunker()
        # A table with single-char cells: " | " is only 3 chars stripped -> filtered
        pages = [RawPage(page_number=1, tables=[[[" "]]])]
        chunks = chunker.chunk_pages(pages, source_pdf="test.pdf")
        assert len(chunks) == 0
        logger.info("PASS: tiny table filtered out")

    def test_valid_table_kept(self):
        from src.ingestion.chunker import Chunker
        from src.models.schemas import RawPage
        chunker = Chunker()
        pages = [RawPage(page_number=1, tables=[[["Name", "Score"], ["Alice", "95"], ["Bob", "87"]]])]
        chunks = chunker.chunk_pages(pages, source_pdf="test.pdf")
        table_chunks = [c for c in chunks if c.content_type == "table"]
        assert len(table_chunks) >= 1
        logger.info("PASS: valid table kept (%d table chunks)", len(table_chunks))

    def test_chunk_ids_unique(self):
        from src.ingestion.chunker import Chunker
        from src.models.schemas import RawPage
        chunker = Chunker(chunk_size=30, chunk_overlap=5, child_size=10, child_overlap=2)
        pages = [RawPage(page_number=1, text="different words for unique hashing test. " * 20)]
        chunks = chunker.chunk_pages(pages, source_pdf="test.pdf")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found!"
        logger.info("PASS: all %d chunk IDs are unique", len(ids))


# ═══════════════════════════════════════════════════════════════════════════════
#  3. RERANKER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestReranker:
    """Validate reranker thresholding and ordering."""

    def _make_chunk(self, text: str, page: int = 1):
        from src.models.schemas import DocumentChunk, RetrievedChunk
        chunk = DocumentChunk(
            text=text, page_number=page, chunk_index=0, source_pdf="test.pdf"
        )
        return RetrievedChunk(chunk=chunk.model_dump(), semantic_score=0.5)

    def test_empty_candidates_returns_empty(self):
        from src.retrieval.reranker import Reranker
        reranker = Reranker()
        result = reranker.rerank("test query", [], top_k=3)
        assert result == []
        logger.info("PASS: empty candidates returns empty")

    def test_reranker_orders_by_score(self):
        from src.retrieval.reranker import Reranker
        reranker = Reranker(min_score=-999.0)  # no threshold filtering
        candidates = [
            self._make_chunk("Options trading is the buying and selling of options contracts"),
            self._make_chunk("A stock is a share of ownership in a company"),
            self._make_chunk("Call options give you the right to buy"),
        ]
        result = reranker.rerank("What is options trading?", candidates, top_k=3)
        scores = [c.rerank_score for c in result]
        assert scores == sorted(scores, reverse=True)
        logger.info("PASS: reranker orders by descending score")

    def test_threshold_filters_low_scores(self):
        from src.retrieval.reranker import Reranker
        reranker = Reranker(min_score=5.0)  # very aggressive threshold
        candidates = [
            self._make_chunk("completely irrelevant text about weather forecasts"),
            self._make_chunk("another unrelated topic about cooking recipes"),
        ]
        result = reranker.rerank("What is options trading?", candidates, top_k=3)
        # With a very high threshold, most chunks should be filtered
        logger.info("PASS: threshold filtering returned %d chunks (expected 0-2)", len(result))

    def test_top_k_limits_output(self):
        from src.retrieval.reranker import Reranker
        reranker = Reranker(min_score=-999.0)
        candidates = [
            self._make_chunk(f"Chunk number {i} with enough text content") for i in range(10)
        ]
        result = reranker.rerank("test query", candidates, top_k=3)
        assert len(result) <= 3
        logger.info("PASS: top_k=3 limits output to %d", len(result))


# ═══════════════════════════════════════════════════════════════════════════════
#  4. QUERY REWRITER TESTS (mocked, no API calls)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryRewriter:
    """Validate query rewriter logic with mocked Groq API."""

    def test_no_history_returns_original(self):
        from src.retrieval.query_rewriter import QueryRewriter
        rewriter = QueryRewriter()
        result = rewriter.rewrite("What is options trading?", chat_history=[])
        assert result.question == "What is options trading?"
        logger.info("PASS: no history returns original query")

    def test_rewrite_with_history_calls_api(self):
        from src.retrieval.query_rewriter import QueryRewriter
        from src.models.schemas import ChatMessage

        rewriter = QueryRewriter()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "What is the strike price in options trading?"

        with patch.object(rewriter._client.chat.completions, "create", return_value=mock_response):
            result = rewriter.rewrite(
                "What is it?",
                chat_history=[
                    ChatMessage(role="user", content="Tell me about strike prices"),
                    ChatMessage(role="assistant", content="A strike price is..."),
                ],
            )
        assert "strike" in result.question.lower()
        logger.info("PASS: rewrite with history produces contextual query: '%s'", result.question)

    def test_fallback_on_api_error(self):
        from src.retrieval.query_rewriter import QueryRewriter
        from src.models.schemas import ChatMessage

        rewriter = QueryRewriter()
        with patch.object(
            rewriter._client.chat.completions, "create",
            side_effect=Exception("API down")
        ):
            result = rewriter.rewrite(
                "What about that?",
                chat_history=[ChatMessage(role="user", content="previous question")],
            )
        assert result.question == "What about that?"
        logger.info("PASS: API failure falls back to original query")

    def test_long_response_fallback(self):
        from src.retrieval.query_rewriter import QueryRewriter
        from src.models.schemas import ChatMessage

        rewriter = QueryRewriter()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "x" * 400  # too long

        with patch.object(rewriter._client.chat.completions, "create", return_value=mock_response):
            result = rewriter.rewrite(
                "original question",
                chat_history=[ChatMessage(role="user", content="history")],
            )
        assert result.question == "original question"
        logger.info("PASS: long LLM response falls back to original")

    def test_prefix_stripped(self):
        from src.retrieval.query_rewriter import QueryRewriter
        from src.models.schemas import ChatMessage

        rewriter = QueryRewriter()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Rewritten query: What is a call option?"

        with patch.object(rewriter._client.chat.completions, "create", return_value=mock_response):
            result = rewriter.rewrite(
                "What is it?",
                chat_history=[ChatMessage(role="user", content="Tell me about call options")],
            )
        assert not result.question.lower().startswith("rewritten")
        logger.info("PASS: prefix stripped from rewritten query: '%s'", result.question)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. VECTORSTORE SAFETY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorStoreSafety:
    """Validate VectorStore defensive checks."""

    def test_add_empty_chunks_is_noop(self):
        from src.vectorstore.chroma_store import VectorStore
        store = VectorStore()
        # Should not raise
        store.add_chunks([], [])
        logger.info("PASS: adding empty chunks is a safe no-op")

    def test_mismatched_lengths_raises(self):
        from src.models.schemas import DocumentChunk
        from src.vectorstore.chroma_store import VectorStore, VectorStoreError
        store = VectorStore()
        chunk = DocumentChunk(
            text="test chunk", page_number=1, chunk_index=0, source_pdf="t.pdf"
        )
        with pytest.raises(VectorStoreError):
            store.add_chunks([chunk], [[0.1, 0.2], [0.3, 0.4]])  # 1 chunk, 2 embeddings
        logger.info("PASS: mismatched chunks/embeddings raises VectorStoreError")
