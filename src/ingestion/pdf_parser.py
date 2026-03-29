"""pdf_parser.py
─────────────
Extracts text, tables, and images from a PDF file.
Handles both digital PDFs and falls back to OCR for scanned pages.

Returns a list[RawPage] — one per page.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from src.models.schemas import RawPage
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum text length below which we attempt OCR on the page
_OCR_FALLBACK_THRESHOLD = 30


class PDFParserError(Exception):
    """Raised when the PDF cannot be parsed."""


class PDFParser:
    """Extracts content from a PDF.

    Usage:
        parser = PDFParser(pdf_path)
        pages  = parser.parse()
    """

    def __init__(self, pdf_path: str | Path, image_output_dir: str | Path = "data/images") -> None:
        self.pdf_path = Path(pdf_path)
        self.image_dir = Path(image_output_dir)

        if not self.pdf_path.exists():
            raise PDFParserError(f"PDF not found: {self.pdf_path}")
        if self.pdf_path.suffix.lower() != ".pdf":
            raise PDFParserError(f"Not a PDF file: {self.pdf_path}")

        self.image_dir.mkdir(parents=True, exist_ok=True)
        logger.info("PDFParser initialised for: %s", self.pdf_path.name)

    # ── Public API ────────────────────────────────────────────────────────────
    def parse(self) -> list[RawPage]:
        """Parse the entire PDF and return a list of RawPage objects."""
        logger.info("Parsing PDF: %s", self.pdf_path.name)
        pages: list[RawPage] = []

        try:
            with pdfplumber.open(self.pdf_path) as plumber_doc, fitz.open(str(self.pdf_path)) as fitz_doc:
                total_pages = len(plumber_doc.pages)
                logger.info("Total pages: %d", total_pages)

                for page_idx in range(total_pages):
                    raw_page = self._parse_page(
                        plumber_page=plumber_doc.pages[page_idx],
                        fitz_page=fitz_doc[page_idx],
                        page_number=page_idx + 1,
                    )
                    pages.append(raw_page)

                    if (page_idx + 1) % 10 == 0 or (page_idx + 1) == total_pages:
                        logger.debug("Parsed %d/%d pages", page_idx + 1, total_pages)

        except PDFParserError:
            raise
        except Exception as exc:
            logger.error("Unexpected error parsing PDF: %s", exc, exc_info=True)
            raise PDFParserError(f"Failed to parse {self.pdf_path.name}: {exc}") from exc

        logger.info(
            "Parsing complete — %d pages, %d with content",
            len(pages),
            sum(1 for p in pages if p.has_content),
        )
        return pages

    def iter_pages(self) -> Generator[RawPage, None, None]:
        """Memory-efficient generator version of parse()."""
        with pdfplumber.open(self.pdf_path) as plumber_doc, fitz.open(str(self.pdf_path)) as fitz_doc:
            for page_idx, plumber_page in enumerate(plumber_doc.pages):
                yield self._parse_page(
                    plumber_page=plumber_page,
                    fitz_page=fitz_doc[page_idx],
                    page_number=page_idx + 1,
                )

    # ── Private helpers ───────────────────────────────────────────────────────
    def _parse_page(
        self,
        plumber_page,
        fitz_page,
        page_number: int,
    ) -> RawPage:
        text = self._extract_text(plumber_page, fitz_page, page_number)
        tables = self._extract_tables(plumber_page, page_number)
        img_paths = self._extract_images(fitz_page, page_number)

        return RawPage(
            page_number=page_number,
            text=text,
            tables=tables,
            image_paths=img_paths,
        )

    def _extract_text(self, plumber_page, fitz_page, page_number: int) -> str:
        """Extract text; fallback to OCR if page appears to be a scan."""
        try:
            text = plumber_page.extract_text() or ""
        except Exception as exc:
            logger.warning("pdfplumber text extraction failed on p%d: %s", page_number, exc)
            text = ""

        # OCR fallback for scanned/image-only pages
        if len(text.strip()) < _OCR_FALLBACK_THRESHOLD:
            text = self._ocr_page(fitz_page, page_number) or text

        return text.strip()

    def _ocr_page(self, fitz_page, page_number: int) -> str:
        """Render page to image and run Tesseract OCR."""
        try:
            import pytesseract

            mat = fitz.Matrix(2, 2)  # 2× zoom → better OCR
            pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="eng")
            logger.debug("OCR fallback used for page %d", page_number)
            return text
        except ImportError:
            logger.warning("pytesseract not installed — OCR skipped for page %d", page_number)
            return ""
        except Exception as exc:
            logger.warning("OCR failed on page %d: %s", page_number, exc)
            return ""

    def _extract_tables(
        self, plumber_page, page_number: int
    ) -> list[list[list[str]]]:
        """Extract tables as nested lists. Empty cells become empty strings."""
        tables: list[list[list[str]]] = []
        try:
            raw_tables = plumber_page.extract_tables() or []
            for table in raw_tables:
                cleaned = [
                    [cell if cell is not None else "" for cell in row]
                    for row in table
                    if row
                ]
                if cleaned:
                    tables.append(cleaned)
        except Exception as exc:
            logger.warning("Table extraction failed on p%d: %s", page_number, exc)
        return tables

    def _extract_images(self, fitz_page, page_number: int) -> list[Path]:
        """Extract embedded images from the page and save as PNG files."""
        saved: list[Path] = []
        try:
            image_list = fitz_page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = fitz_page.parent.extract_image(xref)
                img_bytes = base_image.get("image", b"")

                if len(img_bytes) < 1000:  # skip tiny/icon images
                    continue

                # Stable filename based on content hash (avoids duplicates)
                img_hash = hashlib.md5(img_bytes).hexdigest()[:10]
                out_path = self.image_dir / f"p{page_number:03d}_{img_index:02d}_{img_hash}.png"

                if not out_path.exists():
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    pil_img.save(out_path, "PNG")

                saved.append(out_path)
        except Exception as exc:
            logger.warning("Image extraction failed on p%d: %s", page_number, exc)
        return saved
