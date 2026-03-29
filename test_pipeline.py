"""
tests/test_pipeline.py
───────────────────────
Unit tests for the core pipeline components.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Schema / Pydantic Tests ────────────────────────────────────────────────

class TestDocumentChunk:
    def test_valid_chunk(self):
        from src.models.schemas import DocumentChunk
        chunk = DocumentChunk(
            text="This is a valid test chunk with enough content.",
            page_number=1,
            chunk_index=0,
            source_pdf="test.pdf",
        )
        assert chunk.char_count == len(chunk.text)
        assert chunk.chunk_id != ""

    def test_text_too_short_raises(self):
        from src.models.schemas import DocumentChunk
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DocumentChunk(text="hi", page_number=1, chunk_index=0, source_pdf="test.pdf")

    def test_whitespace_is_stripped(self):
        from src.models.schemas import DocumentChunk
        chunk = DocumentChunk(
            text="   hello world this is a valid chunk content   ",
            page_number=1,
            chunk_index=0,
            source_pdf="test.pdf",
        )
        assert chunk.text == "hello world this is a valid chunk content"

    def test_to_metadata_all_primitives(self):
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


class TestQueryInput:
    def test_valid_query(self):
        from src.models.schemas import QueryInput
        q = QueryInput(question="What is the main concept?")
        assert q.question == "What is the main concept?"

    def test_too_short_raises(self):
        from src.models.schemas import QueryInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryInput(question="ab")

    def test_whitespace_normalized(self):
        from src.models.schemas import QueryInput
        q = QueryInput(question="  what   is  this  ")
        assert q.question == "what is this"

    def test_too_long_raises(self):
        from src.models.schemas import QueryInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryInput(question="x" * 1001)


class TestParsedPage:
    def test_combined_text_empty(self):
        from src.models.schemas import ParsedPage
        page = ParsedPage(page_number=1)
        assert page.is_empty is True

    def test_combined_text_with_content(self):
        from src.models.schemas import ParsedPage
        page = ParsedPage(
            page_number=1,
            text="Hello world",
            tables=[[["col1", "col2"], ["val1", "val2"]]],
            image_descriptions=["A flowchart showing X"],
        )
        combined = page.combined_text
        assert "Hello world" in combined
        assert "TABLE" in combined
        assert "FIGURE" in combined
        assert page.is_empty is False


class TestIngestionResult:
    def test_success_rate(self):
        from src.models.schemas import IngestionResult
        result = IngestionResult(
            pdf_filename="test.pdf",
            total_pages=10,
            pages_parsed=9,
            total_chunks=50,
            duration_seconds=5.0,
            success=True,
        )
        assert result.success_rate == 90.0

    def test_zero_pages(self):
        from src.models.schemas import IngestionResult
        result = IngestionResult(
            pdf_filename="test.pdf",
            total_pages=0,
            pages_parsed=0,
            total_chunks=0,
            duration_seconds=0.1,
            success=False,
        )
        assert result.success_rate == 0.0


# ── Chunker Tests ──────────────────────────────────────────────────────────

class TestChunker:
    def test_basic_chunking(self):
        from src.ingestion.chunker import Chunker
        from src.models.schemas import ParsedPage

        chunker = Chunker(chunk_size=200, chunk_overlap=30)
        pages = [
            ParsedPage(
                page_number=1,
                text="This is a test paragraph. " * 20,
            )
        ]
        chunks = chunker.chunk_pages(pages, source_pdf="test.pdf")
        assert len(chunks) > 0
        for c in chunks:
            assert c.source_pdf == "test.pdf"
            assert c.page_number == 1
            assert len(c.text) >= 10

    def test_table_gets_table_type(self):
        from src.ingestion.chunker import Chunker
        from src.models.schemas import ParsedPage

        chunker = Chunker()
        pages = [
            ParsedPage(
                page_number=1,
                tables=[[["Name", "Score"], ["Alice", "95"], ["Bob", "87"]]],
            )
        ]
        chunks = chunker.chunk_pages(pages, source_pdf="test.pdf")
        table_chunks = [c for c in chunks if c.content_type == "table"]
        assert len(table_chunks) >= 1

    def test_empty_pages_skipped(self):
        from src.ingestion.chunker import Chunker
        from src.models.schemas import ParsedPage

        chunker = Chunker()
        pages = [ParsedPage(page_number=1)]  # empty
        chunks = chunker.chunk_pages(pages, source_pdf="test.pdf")
        assert len(chunks) == 0


# ── Settings Tests ─────────────────────────────────────────────────────────

class TestSettings:
    def test_overlap_validator(self):
        import os
        import importlib
        # Temporarily set an invalid overlap
        os.environ["CHUNK_SIZE"] = "100"
        os.environ["CHUNK_OVERLAP"] = "200"  # invalid: overlap > chunk_size
        os.environ["GROQ_API_KEY"] = "gsk_fake_key_for_testing_purposes_only"

        from pydantic import ValidationError
        with pytest.raises((ValidationError, Exception)):
            from config import settings as s
            # Force re-read
            from config.settings import Settings
            Settings()

        # Cleanup
        del os.environ["CHUNK_SIZE"]
        del os.environ["CHUNK_OVERLAP"]
