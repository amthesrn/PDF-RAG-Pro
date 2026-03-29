"""Project-wide data models (Pydantic v2).

This module defines all shared types used across ingestion, retrieval, and generation.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────
class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    DIAGRAM = "diagram"


class RetrievalMode(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


# ── Ingestion Models ──────────────────────────────────────────────────────────
class RawPage(BaseModel):
    """Represents a single page extracted from the PDF (raw extraction output)."""

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text: str = Field(default="", description="Extracted plain text")
    tables: list[list[list[str]]] = Field(
        default_factory=list,
        description="Nested list: tables[i][row][col]",
    )
    image_paths: list[Path] = Field(
        default_factory=list,
        description="Paths to extracted images from this page",
    )
    has_content: bool = Field(default=True)

    @model_validator(mode="after")
    def mark_empty(self) -> "RawPage":
        if not self.text.strip() and not self.tables and not self.image_paths:
            self.has_content = False
        return self


class ParsedPage(BaseModel):
    """Post-processed page used by chunker.

    Contains combined text (including tables & images) to simplify chunking.
    """

    page_number: int = Field(..., ge=1)
    text: str = Field(default="")
    tables: list[list[list[str]]] = Field(default_factory=list)
    image_descriptions: list[str] = Field(default_factory=list)

    @property
    def combined_text(self) -> str:
        parts: list[str] = []
        if self.text.strip():
            parts.append(self.text.strip())

        if self.tables:
            for table in self.tables:
                rows = [" | ".join(row) for row in table]
                parts.append("\n".join(rows))

        if self.image_descriptions:
            parts.extend(self.image_descriptions)

        return "\n\n".join(parts).strip()

    @property
    def is_empty(self) -> bool:
        return not bool(self.combined_text.strip())


class ImageDescription(BaseModel):
    """LLaVA-generated description for an extracted image/diagram."""

    image_path: Path
    page_number: int
    description: str = Field(..., min_length=1)
    content_type: ContentType = ContentType.IMAGE


class DocumentChunk(BaseModel):
    """A processed text chunk with metadata and IDs."""

    chunk_id: str = Field(default="", description="Unique identifier")
    chunk_index: int = Field(..., ge=0)
    text: str = Field(..., min_length=1)
    page_number: int = Field(..., ge=1)

    content_type: ContentType = ContentType.TEXT
    section_heading: str = Field(default="")
    section: str = Field(default="unknown")

    char_count: int = Field(default=0)
    token_estimate: int = Field(default=0)

    source_pdf: str = Field(default="")
    parent_text: str = Field(default="")

    @model_validator(mode="after")
    def fill_computed_fields(self) -> "DocumentChunk":
        self.char_count = len(self.text)
        self.token_estimate = len(self.text) // 4

        # Generate a stable chunk_id if not provided.
        if not self.chunk_id:
            base = f"{self.source_pdf}:{self.page_number}:{self.chunk_index}:{self.text[:80]}"
            self.chunk_id = hashlib.md5(base.encode("utf-8")).hexdigest()  # type: ignore[name-defined]

        return self

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    def to_metadata(self) -> dict[str, Any]:
        """Flat metadata dict for ChromaDB (primitive values only)."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "content_type": self.content_type.value,
            "section": self.section,
            "section_heading": self.section_heading,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
            "source_pdf": self.source_pdf,
            "parent_text": self.parent_text,
        }


# ── Retrieval Models ──────────────────────────────────────────────────────────
class RetrievedChunk(BaseModel):
    """A chunk returned by the vector store, with scoring."""

    chunk: DocumentChunk
    semantic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    keyword_score: float = Field(default=0.0, ge=0.0)
    combined_score: float = Field(default=0.0)
    rerank_score: float = Field(default=0.0)


class RetrievalResult(BaseModel):
    """Container for a full retrieval operation result."""

    query: str
    mode: RetrievalMode
    chunks: list[RetrievedChunk]
    total_found: int = Field(default=0)

    @model_validator(mode="after")
    def set_total(self) -> "RetrievalResult":
        self.total_found = len(self.chunks)
        return self


# ── Generation Models ─────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    """Single message in a conversation."""

    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)


class RAGRequest(BaseModel):
    """Input to the RAG pipeline."""

    question: str = Field(..., min_length=3)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=4, ge=1, le=20)
    llm_provider: Literal["groq", "gemini"] = Field(default="groq")

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be blank")
        return v.strip()


class QueryInput(BaseModel):
    """Validated query input for the generator."""

    question: str = Field(..., min_length=3)

    @field_validator("question")
    @classmethod
    def trim(cls, v: str) -> str:
        return " ".join(v.strip().split())


class QueryResponse(BaseModel):
    """Response returned from the LLM generator."""

    question: str
    answer: str
    source_chunks: list[DocumentChunk]
    retrieval_mode: RetrievalMode
    confidence: str
    model_used: str
    processing_time_ms: float
    found_in_document: bool = Field(default=True)


# ── Ingestion Result Model ────────────────────────────────────────────────────
class IngestionResult(BaseModel):
    pdf_filename: str
    total_pages: int
    pages_parsed: int
    total_chunks: int
    chunks_text: int = Field(default=0)
    chunks_table: int = Field(default=0)
    chunks_image: int = Field(default=0)
    failed_pages: list[int] = Field(default_factory=list)
    duration_seconds: float
    success: bool

    @property
    def success_rate(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return round(100.0 * self.pages_parsed / self.total_pages, 2)
