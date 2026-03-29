"""
schemas.py
──────────
All Pydantic v2 data models used across the pipeline.
One file for schemas keeps imports clean and avoids circular deps.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────
class ContentType(str, Enum):
    TEXT    = "text"
    TABLE   = "table"
    IMAGE   = "image"
    DIAGRAM = "diagram"


class RetrievalMode(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD  = "keyword"
    HYBRID   = "hybrid"


# ── Ingestion Models ──────────────────────────────────────────────────────────
class RawPage(BaseModel):
    """Represents a single page extracted from the PDF."""
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text: str        = Field(default="", description="Extracted plain text")
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


class ImageDescription(BaseModel):
    """LLaVA-generated description for an extracted image/diagram."""
    image_path: Path
    page_number: int
    description: str = Field(..., min_length=1)
    content_type: ContentType = ContentType.IMAGE


class DocumentChunk(BaseModel):
    """A single processed, metadata-enriched text chunk ready for embedding."""
    chunk_id:     str         = Field(..., description="Unique identifier")
    text:         str         = Field(..., min_length=1)
    page_number:  int         = Field(..., ge=1)
    content_type: ContentType = ContentType.TEXT
    section:      str         = Field(default="unknown")
    char_count:   int         = Field(default=0)
    token_estimate: int       = Field(default=0)
    source_pdf:   str         = Field(default="")

    @model_validator(mode="after")
    def fill_computed_fields(self) -> "DocumentChunk":
        self.char_count     = len(self.text)
        self.token_estimate = len(self.text) // 4   # rough 4 chars/token
        return self

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    def to_metadata_dict(self) -> dict[str, Any]:
        """Metadata dict for ChromaDB (must be flat, primitive values only)."""
        return {
            "chunk_id":       self.chunk_id,
            "page_number":    self.page_number,
            "content_type":   self.content_type.value,
            "section":        self.section,
            "char_count":     self.char_count,
            "token_estimate": self.token_estimate,
            "source_pdf":     self.source_pdf,
        }


# ── Retrieval Models ──────────────────────────────────────────────────────────
class RetrievedChunk(BaseModel):
    """A chunk returned from the vector store, enriched with a score."""
    chunk:            DocumentChunk
    semantic_score:   float = Field(default=0.0, ge=0.0, le=1.0)
    keyword_score:    float = Field(default=0.0, ge=0.0)
    combined_score:   float = Field(default=0.0)
    rerank_score:     float = Field(default=0.0)


class RetrievalResult(BaseModel):
    """Container for a full retrieval operation result."""
    query:       str
    mode:        RetrievalMode
    chunks:      list[RetrievedChunk]
    total_found: int = Field(default=0)

    @model_validator(mode="after")
    def set_total(self) -> "RetrievalResult":
        self.total_found = len(self.chunks)
        return self


# ── Generation Models ─────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    """Single message in a conversation."""
    role:    str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)


class RAGRequest(BaseModel):
    """Input to the RAG pipeline."""
    question:        str               = Field(..., min_length=3)
    chat_history:    list[ChatMessage] = Field(default_factory=list)
    retrieval_mode:  RetrievalMode     = RetrievalMode.HYBRID
    top_k:           int               = Field(default=4, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be blank")
        return v.strip()


class RAGResponse(BaseModel):
    """Output from the RAG pipeline."""
    question:          str
    answer:            str
    source_chunks:     list[DocumentChunk]
    retrieval_mode:    RetrievalMode
    confidence:        float = Field(default=0.0, ge=0.0, le=1.0)
    found_in_document: bool  = Field(default=True)
    latency_ms:        float = Field(default=0.0)


# ── Evaluation Models ─────────────────────────────────────────────────────────
class EvalSample(BaseModel):
    """One Q&A pair for evaluation."""
    question:       str
    ground_truth:   str
    predicted:      str  = Field(default="")
    contexts:       list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Aggregate evaluation metrics."""
    total_samples:      int
    faithfulness:       float = Field(ge=0.0, le=1.0)
    answer_relevancy:   float = Field(ge=0.0, le=1.0)
    context_recall:     float = Field(ge=0.0, le=1.0)
    context_precision:  float = Field(ge=0.0, le=1.0)
    overall_score:      float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def compute_overall(self) -> "EvalReport":
        self.overall_score = round(
            (
                self.faithfulness
                + self.answer_relevancy
                + self.context_recall
                + self.context_precision
            ) / 4,
            4,
        )
        return self
