"""PDF folder scanning service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class PdfScanResult:
    """Structured metadata collected for a discovered PDF file."""

    file_name: str
    file_path: Path
    file_size_bytes: int
    page_count: int | None
    status: str
    error_message: str | None = None


class PdfScanner:
    """Service for discovering PDF files in a selected folder."""

    def scan(self, folder: Path) -> list[PdfScanResult]:
        """Return PDF scan results for non-recursive files in the selected folder."""
        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Path is not a folder: {folder}")

        pdf_files = sorted(
            (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
            key=lambda path: path.name.lower(),
        )

        return [self._scan_file(file_path) for file_path in pdf_files]

    def _scan_file(self, file_path: Path) -> PdfScanResult:
        file_size = self._safe_file_size(file_path)

        try:
            with pymupdf.open(file_path) as document:
                page_count = len(document)
        except Exception as error:
            return PdfScanResult(
                file_name=file_path.name,
                file_path=file_path.resolve(),
                file_size_bytes=file_size,
                page_count=None,
                status="Unreadable",
                error_message=str(error),
            )

        return PdfScanResult(
            file_name=file_path.name,
            file_path=file_path.resolve(),
            file_size_bytes=file_size,
            page_count=page_count,
            status="Ready",
        )

    def _safe_file_size(self, file_path: Path) -> int:
        try:
            return file_path.stat().st_size
        except OSError:
            return 0
