"""Force reindex selected source PDFs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from app.core.pipeline import IndexingPipeline
from app.services.deletion_service import IndexDeletionService
from app.services.file_fingerprint import FileFingerprintService
from app.storage.document_registry import DocumentRegistry, IndexedDocumentRecord


@dataclass(frozen=True)
class ReindexDocumentResult:
    """Per-document reindex result."""

    document_id: str
    file_name: str
    source_path: str
    status: str
    removed_points: int
    stored_points: int
    error_message: str | None = None


@dataclass(frozen=True)
class ReindexRunSummary:
    """Structured summary for a reindex operation."""

    requested_documents: int
    reindexed_documents: int
    newly_indexed_documents: int
    skipped_documents: int
    failed_documents: int
    removed_points: int
    stored_points: int
    elapsed_seconds: float
    results: list[ReindexDocumentResult]


class ReindexService:
    """Remove old vectors for selected documents and index them again."""

    def __init__(
        self,
        indexing_pipeline: IndexingPipeline,
        deletion_service: IndexDeletionService,
        registry: DocumentRegistry,
        fingerprint_service: FileFingerprintService | None = None,
    ) -> None:
        self.indexing_pipeline = indexing_pipeline
        self.deletion_service = deletion_service
        self.registry = registry
        self.fingerprint_service = fingerprint_service or FileFingerprintService()

    def reindex_pdfs(self, pdf_paths: list[Path]) -> ReindexRunSummary:
        """Force reindex selected PDFs one by one."""
        start_time = perf_counter()
        records_before = self.registry.load_documents()
        results: list[ReindexDocumentResult] = []

        for pdf_path in pdf_paths:
            if not pdf_path.exists() or not pdf_path.is_file():
                results.append(
                    ReindexDocumentResult(
                        document_id="",
                        file_name=pdf_path.name,
                        source_path=str(pdf_path),
                        status="skipped_unavailable",
                        removed_points=0,
                        stored_points=0,
                        error_message="Source file is not available.",
                    )
                )
                continue

            document_id = self._document_id_for_path(pdf_path)
            was_indexed = document_id in records_before
            removed_points = 0

            if was_indexed:
                clear_summary = self.deletion_service.clear_selected([document_id])
                removed_points = clear_summary.removed_points
                failed_clear = next(
                    (detail for detail in clear_summary.details if detail.status == "failed"),
                    None,
                )
                if failed_clear is not None:
                    results.append(
                        ReindexDocumentResult(
                            document_id=document_id,
                            file_name=pdf_path.name,
                            source_path=str(pdf_path),
                            status="failed",
                            removed_points=removed_points,
                            stored_points=0,
                            error_message=failed_clear.error_message,
                        )
                    )
                    continue

            indexing_summary = self.indexing_pipeline.index_pdfs([pdf_path])
            indexing_result = indexing_summary.results[0] if indexing_summary.results else None
            if indexing_result is None or not indexing_result.success:
                self._record_index_error(pdf_path, document_id, indexing_result)
                results.append(
                    ReindexDocumentResult(
                        document_id=document_id,
                        file_name=pdf_path.name,
                        source_path=str(pdf_path),
                        status="failed",
                        removed_points=removed_points,
                        stored_points=0,
                        error_message=indexing_result.error_message if indexing_result else "No indexing result.",
                    )
                )
                continue

            results.append(
                ReindexDocumentResult(
                    document_id=document_id,
                    file_name=pdf_path.name,
                    source_path=str(indexing_result.file_path),
                    status="reindexed" if was_indexed else "newly_indexed",
                    removed_points=removed_points,
                    stored_points=indexing_result.stored_points,
                )
            )

        return ReindexRunSummary(
            requested_documents=len(pdf_paths),
            reindexed_documents=sum(1 for result in results if result.status == "reindexed"),
            newly_indexed_documents=sum(1 for result in results if result.status == "newly_indexed"),
            skipped_documents=sum(1 for result in results if result.status.startswith("skipped")),
            failed_documents=sum(1 for result in results if result.status == "failed"),
            removed_points=sum(result.removed_points for result in results),
            stored_points=sum(result.stored_points for result in results),
            elapsed_seconds=perf_counter() - start_time,
            results=results,
        )

    def export_summary(self, summary: ReindexRunSummary, file_name: str) -> Path:
        """Export reindex summary under shared metadata."""
        self.registry.ensure_directories()
        output_path = self.registry.metadata_dir / file_name
        output_path.write_text(
            json.dumps(reindex_summary_to_json(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output_path

    def _document_id_for_path(self, pdf_path: Path) -> str:
        from app.core.block_builder import make_document_id

        return make_document_id(pdf_path)

    def _record_index_error(self, pdf_path: Path, document_id: str, indexing_result: object) -> None:
        error_message = getattr(indexing_result, "error_message", None) or "Indexing failed."
        try:
            fingerprint = self.fingerprint_service.get_content_fingerprint(pdf_path)
        except OSError:
            fingerprint = None
        self.registry.register_document(
            IndexedDocumentRecord(
                document_id=document_id,
                doc_id=document_id,
                source_path=str(pdf_path.resolve()),
                file_name=pdf_path.name,
                indexed_at="",
                stored_points=0,
                file_size=fingerprint.file_size if fingerprint else 0,
                modified_at=fingerprint.modified_at if fingerprint else "",
                source_fingerprint=fingerprint.source_fingerprint if fingerprint and fingerprint.source_fingerprint else "",
                status="index_error",
                error_message=error_message,
            )
        )


def reindex_summary_to_json(summary: ReindexRunSummary) -> dict[str, object]:
    """Convert a reindex summary to JSON data."""
    return {
        "requested_documents": summary.requested_documents,
        "reindexed_documents": summary.reindexed_documents,
        "newly_indexed_documents": summary.newly_indexed_documents,
        "skipped_documents": summary.skipped_documents,
        "failed_documents": summary.failed_documents,
        "removed_points": summary.removed_points,
        "stored_points": summary.stored_points,
        "elapsed_seconds": round(summary.elapsed_seconds, 3),
        "results": [
            {
                "document_id": result.document_id,
                "file_name": result.file_name,
                "source_path": result.source_path,
                "status": result.status,
                "removed_points": result.removed_points,
                "stored_points": result.stored_points,
                "error_message": result.error_message,
            }
            for result in summary.results
        ],
    }
