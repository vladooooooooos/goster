"""Coordinated shared-index deletion operations."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from app.storage.document_registry import DocumentRegistry, IndexedDocumentRecord
from app.storage.qdrant_store import QdrantStore


@dataclass(frozen=True)
class DocumentDeletionDetail:
    """Per-document clear result shown in logs and summaries."""

    document_id: str
    file_name: str
    source_path: str
    status: str
    removed_points: int
    removed_cache_entries: int
    error_message: str | None = None


@dataclass(frozen=True)
class ClearOperationSummary:
    """Structured result for clear selected and clear all operations."""

    operation: str
    requested_documents: int
    removed_documents: int
    skipped_documents: int
    removed_points: int
    removed_cache_entries: int
    elapsed_seconds: float
    details: list[DocumentDeletionDetail]


class IndexDeletionService:
    """Delete shared index data without touching source PDFs."""

    def __init__(self, qdrant_store: QdrantStore, registry: DocumentRegistry) -> None:
        self.qdrant_store = qdrant_store
        self.registry = registry

    def clear_selected(self, document_ids: list[str]) -> ClearOperationSummary:
        """Remove selected documents from Qdrant, registry, and index-owned artifacts."""
        start_time = perf_counter()
        records = self.registry.load_documents()
        unique_document_ids = unique_nonempty(document_ids)
        qdrant_run = self.qdrant_store.delete_documents(unique_document_ids)
        qdrant_results = {result.document_id: result for result in qdrant_run.results}

        details: list[DocumentDeletionDetail] = []
        removable_document_ids: list[str] = []
        removed_cache_entries = 0

        for document_id in unique_document_ids:
            record = records.get(document_id)
            qdrant_result = qdrant_results.get(document_id)
            removed_points = qdrant_result.removed_points if qdrant_result else 0
            error_message = qdrant_result.error_message if qdrant_result else None

            if error_message:
                details.append(
                    make_detail(
                        document_id=document_id,
                        record=record,
                        status="failed",
                        removed_points=removed_points,
                        removed_cache_entries=0,
                        error_message=error_message,
                    )
                )
                continue

            if record is None and removed_points == 0:
                details.append(
                    make_detail(
                        document_id=document_id,
                        record=None,
                        status="skipped_not_indexed",
                        removed_points=0,
                        removed_cache_entries=0,
                    )
                )
                continue

            removed_for_record = self._remove_record_artifacts(record) if record else 0
            removed_cache_entries += removed_for_record
            removable_document_ids.append(document_id)
            details.append(
                make_detail(
                    document_id=document_id,
                    record=record,
                    status="removed",
                    removed_points=removed_points,
                    removed_cache_entries=removed_for_record,
                )
            )

        if removable_document_ids:
            self.registry.remove_documents(removable_document_ids)

        return ClearOperationSummary(
            operation="clear_selected",
            requested_documents=len(unique_document_ids),
            removed_documents=sum(1 for detail in details if detail.status == "removed"),
            skipped_documents=sum(1 for detail in details if detail.status.startswith("skipped")),
            removed_points=sum(detail.removed_points for detail in details),
            removed_cache_entries=removed_cache_entries,
            elapsed_seconds=perf_counter() - start_time,
            details=details,
        )

    def clear_all(self) -> ClearOperationSummary:
        """Wipe only shared index-owned data and recreate empty directories."""
        start_time = perf_counter()
        records = self.registry.load_documents()
        qdrant_run = self.qdrant_store.clear_all()
        removed_cache_entries = self._clear_indexator_artifact_dir(self.registry.cache_dir)
        removed_cache_entries += self._clear_indexator_artifact_dir(self.registry.debug_dir)
        removed_registry_entries = self.registry.clear()
        self.registry.ensure_directories()

        details = [
            make_detail(
                document_id=record.document_id,
                record=record,
                status="removed",
                removed_points=0,
                removed_cache_entries=0,
            )
            for record in records.values()
        ]

        return ClearOperationSummary(
            operation="clear_all",
            requested_documents=len(records),
            removed_documents=removed_registry_entries,
            skipped_documents=0,
            removed_points=qdrant_run.removed_points,
            removed_cache_entries=removed_cache_entries,
            elapsed_seconds=perf_counter() - start_time,
            details=details,
        )

    def export_summary(self, summary: ClearOperationSummary, file_name: str) -> Path:
        """Export a deletion summary under shared metadata."""
        self.registry.ensure_directories()
        output_path = self.registry.deletion_summary_dir / file_name
        output_path.write_text(
            json.dumps(clear_summary_to_json(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output_path

    def _remove_record_artifacts(self, record: IndexedDocumentRecord | None) -> int:
        if record is None:
            return 0

        removed_count = 0
        for artifact_path in record.artifact_paths:
            path = Path(artifact_path)
            if is_index_owned_artifact(path, self.registry.shared_data_root) and path.exists() and path.is_file():
                path.unlink()
                removed_count += 1
        return removed_count

    def _clear_indexator_artifact_dir(self, directory: Path) -> int:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            return 0

        removed_count = 0
        for child in directory.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
                removed_count += 1
            elif child.is_file():
                child.unlink()
                removed_count += 1
        directory.mkdir(parents=True, exist_ok=True)
        return removed_count


def make_detail(
    document_id: str,
    record: IndexedDocumentRecord | None,
    status: str,
    removed_points: int,
    removed_cache_entries: int,
    error_message: str | None = None,
) -> DocumentDeletionDetail:
    """Build one deletion detail from optional registry data."""
    return DocumentDeletionDetail(
        document_id=document_id,
        file_name=record.file_name if record else "",
        source_path=record.source_path if record else "",
        status=status,
        removed_points=removed_points,
        removed_cache_entries=removed_cache_entries,
        error_message=error_message,
    )


def clear_summary_to_json(summary: ClearOperationSummary) -> dict[str, object]:
    """Convert a clear operation summary to JSON data."""
    return {
        "operation": summary.operation,
        "requested_documents": summary.requested_documents,
        "removed_documents": summary.removed_documents,
        "skipped_documents": summary.skipped_documents,
        "removed_points": summary.removed_points,
        "removed_cache_entries": summary.removed_cache_entries,
        "elapsed_seconds": round(summary.elapsed_seconds, 3),
        "details": [
            {
                "document_id": detail.document_id,
                "file_name": detail.file_name,
                "source_path": detail.source_path,
                "status": detail.status,
                "removed_points": detail.removed_points,
                "removed_cache_entries": detail.removed_cache_entries,
                "error_message": detail.error_message,
            }
            for detail in summary.details
        ],
    }


def is_index_owned_artifact(path: Path, shared_data_root: Path) -> bool:
    """Return whether a path is inside shared index-owned cache/debug storage."""
    try:
        resolved_path = path.resolve()
        resolved_root = shared_data_root.resolve()
    except OSError:
        return False

    return resolved_path.is_relative_to(resolved_root / "cache" / "indexator") or resolved_path.is_relative_to(
        resolved_root / "debug" / "indexator"
    )


def unique_nonempty(values: list[str]) -> list[str]:
    """Return unique non-empty strings preserving input order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
