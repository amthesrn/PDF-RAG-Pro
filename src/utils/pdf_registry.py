"""PDF Registry — tracks previously uploaded PDFs to avoid re-indexing.

Uses a JSON file (`data/pdf_registry.json`) to map content hashes to
PDF metadata. This allows the system to skip indexing if the same PDF
is uploaded again.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import settings


_REGISTRY_PATH = settings.data_dir / "pdf_registry.json"


class PDFRegistry:
    """Manages a persistent registry of indexed PDFs."""

    def __init__(self) -> None:
        self._path = _REGISTRY_PATH
        self._data: dict = self._load()

    # ── Public API ────────────────────────────────────────────────────────

    @staticmethod
    def compute_hash(file_bytes: bytes) -> str:
        """Return an MD5 hex digest of the file content."""
        return hashlib.md5(file_bytes).hexdigest()

    def is_indexed(self, file_hash: str) -> bool:
        """Check whether a PDF with this hash was previously indexed."""
        return file_hash in self._data

    def get_entry(self, file_hash: str) -> Optional[dict]:
        """Return registry entry for a given hash, or None."""
        return self._data.get(file_hash)

    def register(
        self,
        file_hash: str,
        filename: str,
        total_pages: int,
        total_chunks: int,
    ) -> None:
        """Record a newly indexed PDF."""
        self._data[file_hash] = {
            "filename": filename,
            "file_hash": file_hash,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def get_all(self) -> list[dict]:
        """Return a list of all registered PDFs (most recent first)."""
        entries = list(self._data.values())
        entries.sort(key=lambda e: e.get("indexed_at", ""), reverse=True)
        return entries

    def remove(self, file_hash: str) -> bool:
        """Remove a PDF entry from the registry. Returns True if found."""
        if file_hash in self._data:
            del self._data[file_hash]
            self._save()
            return True
        return False

    # ── Private helpers ───────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
