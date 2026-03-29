"""Chunker for PDF pages.

Splits page text into overlapping chunks and tags tables/images.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional

from config.settings import settings
from src.models.schemas import DocumentChunk, ImageDescription, RawPage


def _make_chunk_id(source_pdf: str, page_number: int, chunk_index: int, text: str) -> str:
    base = f"{source_pdf}:{page_number}:{chunk_index}:{text[:80]}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


class Chunker:
    """Chunker that splits PDF pages into textual chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        child_size: int = 100,
        child_overlap: int = 20,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk_pages(
        self,
        pages: Iterable[RawPage],
        source_pdf: str,
        image_descriptions: Optional[List[ImageDescription]] = None,
    ) -> list[DocumentChunk]:
        """Convert raw pages into a list of DocumentChunk objects."""
        chunks: list[DocumentChunk] = []
        image_map = self._group_image_descriptions(image_descriptions or [])

        for page in pages:
            if not page.has_content:
                continue

            # Text chunks
            text = page.text.strip()
            if text:
                chunks.extend(
                    self._chunk_text(
                        text=text,
                        page_number=page.page_number,
                        source_pdf=source_pdf,
                    )
                )

            # Table chunks
            for table in page.tables:
                if not table:
                    continue
                table_text = "\n".join(" | ".join(row) for row in table)
                if len(table_text.strip()) < 5:
                    continue
                
                idx = len(chunks)
                chunks.append(
                    DocumentChunk(
                        chunk_id=_make_chunk_id(source_pdf, page.page_number, idx, table_text),
                        chunk_index=idx,
                        text=table_text,
                        page_number=page.page_number,
                        content_type="table",
                        section_heading="",
                        source_pdf=source_pdf,
                        parent_text=table_text,
                    )
                )

            # Image description chunks
            for img_desc in image_map.get(page.page_number, []):
                idx = len(chunks)
                chunks.append(
                    DocumentChunk(
                        chunk_id=_make_chunk_id(source_pdf, page.page_number, idx, img_desc.description),
                        chunk_index=idx,
                        text=img_desc.description,
                        page_number=page.page_number,
                        content_type="image",
                        section_heading="",
                        source_pdf=source_pdf,
                        parent_text=img_desc.description,
                    )
                )

        return chunks

    def _chunk_text(self, text: str, page_number: int, source_pdf: str) -> list[DocumentChunk]:
        """Split text into parent chunks, then into intersecting child chunks."""
        words = text.split()
        if not words:
            return []

        chunks: list[DocumentChunk] = []
        parent_start = 0
        chunk_idx = 0

        while parent_start < len(words):
            parent_end = min(parent_start + self.chunk_size, len(words))
            parent_words = words[parent_start:parent_end]
            parent_text = " ".join(parent_words).strip()

            if parent_text:
                # Split parent into smaller children for tight vector indexing
                child_start = 0
                while child_start < len(parent_words):
                    child_end = min(child_start + self.child_size, len(parent_words))
                    child_words = parent_words[child_start:child_end]
                    child_text = " ".join(child_words).strip()
                    
                    if child_text:
                        chunk_id = _make_chunk_id(source_pdf, page_number, chunk_idx, child_text)
                        chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                chunk_index=chunk_idx,
                                text=child_text,
                                page_number=page_number,
                                content_type="text",
                                section_heading="",
                                source_pdf=source_pdf,
                                parent_text=parent_text,
                            )
                        )
                        chunk_idx += 1
                    
                    if child_end == len(parent_words):
                        break
                    child_start = child_end - self.child_overlap
                    if child_start < 0:
                        child_start = 0

            if parent_end == len(words):
                break

            parent_start = parent_end - self.chunk_overlap
            if parent_start < 0:
                parent_start = 0

        return chunks

    @staticmethod
    def _group_image_descriptions(image_descriptions: List[ImageDescription]) -> dict[int, List[ImageDescription]]:
        grouped: dict[int, List[ImageDescription]] = {}
        for desc in image_descriptions:
            grouped.setdefault(desc.page_number, []).append(desc)
        return grouped


# Alias for compatibility
DocumentChunker = Chunker
