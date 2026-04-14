"""Resolve UI indexed state from source scan results and shared registry records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.block_builder import make_document_id
from app.services.file_fingerprint import FileFingerprintService
from app.services.pdf_scanner import PdfScanResult
from app.storage.document_registry import IndexedDocumentRecord


READY = "Ready"
INDEXED = "Indexed"
INDEXED_STALE = "Indexed (stale)"
MISSING_SOURCE = "Missing source"
INDEX_ERROR = "Index error"


@dataclass(frozen=True)
class IndexedFileState:
    """Merged source/registry state for one UI table row."""

    file_name: str
    file_path: Path
    file_size_bytes: int
    page_count: int | None
    status: str
    document_id: str
    error_message: str | None = None


class IndexedStateResolver:
    """Classify current source PDFs against shared registry state."""

    def __init__(self, fingerprint_service: FileFingerprintService) -> None:
        self.fingerprint_service = fingerprint_service

    def resolve_scan_results(
        self,
        scan_results: list[PdfScanResult],
        records: dict[str, IndexedDocumentRecord],
        scanned_folder: Path | None,
    ) -> list[IndexedFileState]:
        """Return UI states for scanned files plus missing-source records in the folder."""
        states: list[IndexedFileState] = []
        seen_document_ids: set[str] = set()

        for result in scan_results:
            document_id = make_document_id(result.file_path)
            seen_document_ids.add(document_id)
            record = records.get(document_id)
            states.append(self._resolve_scan_result(result, document_id, record))

        if scanned_folder is not None:
            scanned_folder_resolved = scanned_folder.resolve()
            for record in records.values():
                if record.document_id in seen_document_ids:
                    continue
                source_path = Path(record.source_path)
                if source_path.exists():
                    continue
                try:
                    if source_path.parent.resolve() != scanned_folder_resolved:
                        continue
                except OSError:
                    continue
                states.append(
                    IndexedFileState(
                        file_name=record.file_name,
                        file_path=source_path,
                        file_size_bytes=record.file_size,
                        page_count=None,
                        status=MISSING_SOURCE if record.status != "index_error" else INDEX_ERROR,
                        document_id=record.document_id,
                        error_message=record.error_message,
                    )
                )

        return states

    def _resolve_scan_result(
        self,
        result: PdfScanResult,
        document_id: str,
        record: IndexedDocumentRecord | None,
    ) -> IndexedFileState:
        if result.status != READY:
            return IndexedFileState(
                file_name=result.file_name,
                file_path=result.file_path,
                file_size_bytes=result.file_size_bytes,
                page_count=result.page_count,
                status=result.status,
                document_id=document_id,
                error_message=result.error_message,
            )

        if record is None:
            return IndexedFileState(
                file_name=result.file_name,
                file_path=result.file_path,
                file_size_bytes=result.file_size_bytes,
                page_count=result.page_count,
                status=READY,
                document_id=document_id,
            )

        if record.status == "index_error":
            return IndexedFileState(
                file_name=result.file_name,
                file_path=result.file_path,
                file_size_bytes=result.file_size_bytes,
                page_count=result.page_count,
                status=INDEX_ERROR,
                document_id=document_id,
                error_message=record.error_message,
            )

        try:
            metadata = self.fingerprint_service.get_metadata(result.file_path)
        except OSError as error:
            return IndexedFileState(
                file_name=result.file_name,
                file_path=result.file_path,
                file_size_bytes=result.file_size_bytes,
                page_count=result.page_count,
                status=MISSING_SOURCE,
                document_id=document_id,
                error_message=str(error),
            )

        is_stale = (
            record.file_size <= 0
            or not record.modified_at
            or record.file_size != metadata.file_size
            or record.modified_at != metadata.modified_at
        )
        return IndexedFileState(
            file_name=result.file_name,
            file_path=result.file_path,
            file_size_bytes=result.file_size_bytes,
            page_count=result.page_count,
            status=INDEXED_STALE if is_stale else INDEXED,
            document_id=document_id,
        )
