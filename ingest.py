"""
scripts/ingest.py
──────────────────
CLI entry point to ingest one or all PDFs from the data/pdfs/ folder.

Usage:
    # Ingest all PDFs in data/pdfs/
    python scripts/ingest.py

    # Ingest a specific PDF
    python scripts/ingest.py --pdf data/pdfs/your_document.pdf

    # Wipe existing collection first (re-ingest from scratch)
    python scripts/ingest.py --reset
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from loguru import logger

# Bootstrap logger before any other import
from src.utils.logger import setup_logger
setup_logger()

from config.settings import settings
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.image_processor import ImageProcessor
from src.ingestion.chunker import Chunker
from src.embeddings.embedder import Embedder
from src.vectorstore.chroma_store import VectorStore
from src.models.schemas import ImageDescription, IngestionResult


def ingest_pdf(
    pdf_path: Path,
    chunker: Chunker,
    embedder: Embedder,
    vector_store: VectorStore,
    image_processor: ImageProcessor,
) -> IngestionResult:
    """
    Full pipeline for a single PDF file.
    Returns an IngestionResult summary.
    """
    start = time.perf_counter()
    logger.info(f"{'='*60}")
    logger.info(f"Ingesting: {pdf_path.name}")
    logger.info(f"{'='*60}")

    failed_pages: list[int] = []
    chunks_text = chunks_table = chunks_image = 0

    # ── Step 1: Parse PDF ────────────────────────────────────────────
    try:
        parser = PDFParser(pdf_path, image_output_dir=str(settings.data_dir / "images"))
        pages = parser.parse()
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Parse failed for '{pdf_path.name}': {exc}")
        return IngestionResult(
            pdf_filename=pdf_path.name,
            total_pages=0,
            pages_parsed=0,
            total_chunks=0,
            duration_seconds=round(time.perf_counter() - start, 2),
            success=False,
            error_message=str(exc),
        )

    failed_pages = [p.page_number for p in pages if not p.has_content]
    good_pages = [p for p in pages if p.has_content]
    logger.info(f"Pages parsed: {len(good_pages)}/{len(pages)} usable")

    # Image descriptions
    img_descs: list[ImageDescription] = []
    for page in good_pages:
        for img_path in page.image_paths:
            try:
                desc = image_processor.describe(img_path.read_bytes(), page.page_number)
                if desc:
                    img_descs.append(
                        ImageDescription(
                            image_path=img_path,
                            page_number=page.page_number,
                            description=desc,
                        )
                    )
            except Exception as exc:
                logger.warning("Failed to describe image %s: %s", img_path, exc)

    # ── Step 2: Chunk ────────────────────────────────────────────────
    try:
        chunks = chunker.chunk_pages(
            good_pages, source_pdf=pdf_path.name, image_descriptions=img_descs
        )
    except Exception as exc:
        logger.error(f"Chunking failed for '{pdf_path.name}': {exc}")
        return IngestionResult(
            pdf_filename=pdf_path.name,
            total_pages=len(pages),
            pages_parsed=len(good_pages),
            total_chunks=0,
            failed_pages=failed_pages,
            duration_seconds=round(time.perf_counter() - start, 2),
            success=False,
            error_message=str(exc),
        )

    chunks_text  = sum(1 for c in chunks if c.content_type == "text")
    chunks_table = sum(1 for c in chunks if c.content_type == "table")
    chunks_image = sum(1 for c in chunks if c.content_type == "image")

    logger.info(
        f"Chunks: {len(chunks)} total "
        f"({chunks_text} text | {chunks_table} table | {chunks_image} image)"
    )

    if not chunks:
        logger.warning(f"No chunks produced from '{pdf_path.name}'. Skipping.")
        return IngestionResult(
            pdf_filename=pdf_path.name,
            total_pages=len(pages),
            pages_parsed=len(good_pages),
            total_chunks=0,
            failed_pages=failed_pages,
            duration_seconds=round(time.perf_counter() - start, 2),
            success=False,
            error_message="No chunks produced.",
        )

    # ── Step 3: Embed ────────────────────────────────────────────────
    try:
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_documents(texts)
    except RuntimeError as exc:
        logger.error(f"Embedding failed for '{pdf_path.name}': {exc}")
        return IngestionResult(
            pdf_filename=pdf_path.name,
            total_pages=len(pages),
            pages_parsed=len(good_pages),
            total_chunks=len(chunks),
            failed_pages=failed_pages,
            duration_seconds=round(time.perf_counter() - start, 2),
            success=False,
            error_message=str(exc),
        )

    # ── Step 4: Store in ChromaDB ────────────────────────────────────
    try:
        vector_store.add_chunks(chunks, embeddings)
    except RuntimeError as exc:
        logger.error(f"Vector store failed for '{pdf_path.name}': {exc}")
        return IngestionResult(
            pdf_filename=pdf_path.name,
            total_pages=len(pages),
            pages_parsed=len(good_pages),
            total_chunks=len(chunks),
            failed_pages=failed_pages,
            duration_seconds=round(time.perf_counter() - start, 2),
            success=False,
            error_message=str(exc),
        )

    duration = round(time.perf_counter() - start, 2)
    result = IngestionResult(
        pdf_filename=pdf_path.name,
        total_pages=len(pages),
        pages_parsed=len(good_pages),
        total_chunks=len(chunks),
        chunks_text=chunks_text,
        chunks_table=chunks_table,
        chunks_image=chunks_image,
        failed_pages=failed_pages,
        duration_seconds=duration,
        success=True,
    )

    logger.success(
        f"✓ '{pdf_path.name}' ingested in {duration}s | "
        f"{len(chunks)} chunks | success_rate={result.success_rate}%"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG vector store.")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Path to a specific PDF. If not provided, ingests all PDFs in data/pdfs/",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing vector store collection before ingesting.",
    )
    args = parser.parse_args()

    # ── Collect PDF paths ────────────────────────────────────────────
    if args.pdf:
        pdf_paths = [args.pdf]
    else:
        pdf_paths = list(settings.pdf_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.error(
                f"No PDFs found in '{settings.pdf_dir}'. "
                f"Place your PDF(s) there and run again."
            )
            return

    logger.info(f"PDFs to ingest: {[p.name for p in pdf_paths]}")

    # ── Initialise pipeline components ──────────────────────────────
    img_processor = ImageProcessor()

    chunker      = Chunker()
    embedder     = Embedder()
    vector_store = VectorStore()

    if args.reset:
        logger.warning("--reset flag: deleting existing collection…")
        vector_store.delete_collection()

    # ── Run ingestion ────────────────────────────────────────────────
    all_results: list[IngestionResult] = []
    for pdf_path in pdf_paths:
        result = ingest_pdf(pdf_path, chunker, embedder, vector_store, img_processor)
        all_results.append(result)

    # ── Final summary ────────────────────────────────────────────────
    success_count = sum(1 for r in all_results if r.success)
    total_chunks  = sum(r.total_chunks for r in all_results)

    logger.info(f"\n{'='*60}")
    logger.info(f"INGESTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"PDFs processed : {len(all_results)}")
    logger.info(f"Successful     : {success_count}")
    logger.info(f"Failed         : {len(all_results) - success_count}")
    logger.info(f"Total chunks   : {total_chunks}")
    logger.info(f"Vector store   : {vector_store.count()} total chunks stored")
    logger.info(f"{'='*60}")

    if success_count == len(all_results):
        logger.success("All PDFs ingested successfully! Run: streamlit run app/streamlit_app.py")
    else:
        for r in all_results:
            if not r.success:
                logger.error(f"  FAILED: {r.pdf_filename} — {r.error_message}")


if __name__ == "__main__":
    main()
