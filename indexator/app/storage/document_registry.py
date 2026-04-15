"""Shared document registry for indexed PDF state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REGISTRY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexedDocumentRecord:
    """One document entry persisted in the shared index registry."""

    document_id: str
    source_path: str
    file_name: str
    indexed_at: str
    stored_points: int
    file_size: int
    modified_at: str
    source_fingerprint: str
    status: str = "indexed"
    doc_id: str | None = None
    error_message: str | None = None
    artifact_paths: list[str] = field(default_factory=list)


class DocumentRegistry:
    """Read and write indexed document metadata under shared storage."""

    def __init__(self, shared_data_root: Path) -> None:
        self.shared_data_root = shared_data_root
        self.metadata_dir = shared_data_root / "metadata"
        self.cache_dir = shared_data_root / "cache" / "indexator"
        self.debug_dir = shared_data_root / "debug" / "indexator"
        self.deletion_summary_dir = self.metadata_dir / "deletion_summaries"
        self.documents_path = self.metadata_dir / "documents.json"
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create deterministic shared index-owned directories."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.deletion_summary_dir.mkdir(parents=True, exist_ok=True)

    def load_documents(self) -> dict[str, IndexedDocumentRecord]:
        """Load indexed document records keyed by document id."""
        if not self.documents_path.exists():
            return {}

        try:
            payload = json.loads(self.documents_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        records: dict[str, IndexedDocumentRecord] = {}
        for raw_document in payload.get("documents", []):
            record = record_from_json(raw_document)
            if record:
                records[record.document_id] = record
        return records

    def save_documents(self, records: dict[str, IndexedDocumentRecord]) -> None:
        """Persist document records in a stable order."""
        self.ensure_directories()
        payload = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "documents": [
                record_to_json(record)
                for record in sorted(records.values(), key=lambda item: item.file_name.lower())
            ],
        }
        self.documents_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def register_document(self, record: IndexedDocumentRecord) -> None:
        """Add or replace one indexed document record."""
        records = self.load_documents()
        records[record.document_id] = record
        self.save_documents(records)

    def get_document(self, document_id: str) -> IndexedDocumentRecord | None:
        """Return one document record when present."""
        return self.load_documents().get(document_id)

    def remove_documents(self, document_ids: list[str]) -> list[IndexedDocumentRecord]:
        """Remove records and return the entries that existed."""
        records = self.load_documents()
        removed: list[IndexedDocumentRecord] = []
        for document_id in document_ids:
            record = records.pop(document_id, None)
            if record:
                removed.append(record)
        self.save_documents(records)
        return removed

    def clear(self) -> int:
        """Clear the registry and return the number of removed entries."""
        removed_count = len(self.load_documents())
        self.save_documents({})
        return removed_count


def record_to_json(record: IndexedDocumentRecord) -> dict[str, Any]:
    """Convert a registry record to JSON data."""
    return {
        "document_id": record.document_id,
        "doc_id": record.doc_id or record.document_id,
        "source_path": record.source_path,
        "file_name": record.file_name,
        "indexed_at": record.indexed_at,
        "stored_points": record.stored_points,
        "file_size": record.file_size,
        "modified_at": record.modified_at,
        "source_fingerprint": record.source_fingerprint,
        "status": record.status,
        "error_message": record.error_message,
        "artifact_paths": list(record.artifact_paths),
    }


def record_from_json(payload: Any) -> IndexedDocumentRecord | None:
    """Parse one registry record defensively."""
    if not isinstance(payload, dict):
        return None

    document_id = string_value(payload.get("document_id")) or string_value(payload.get("doc_id"))
    source_path = string_value(payload.get("source_path")) or string_value(payload.get("file_path"))
    file_name = string_value(payload.get("file_name")) or string_value(payload.get("filename"))
    indexed_at = string_value(payload.get("indexed_at"))
    if not document_id or not source_path or not file_name:
        return None

    artifact_paths = payload.get("artifact_paths")
    return IndexedDocumentRecord(
        document_id=document_id,
        doc_id=string_value(payload.get("doc_id")) or document_id,
        source_path=source_path,
        file_name=file_name,
        indexed_at=indexed_at,
        stored_points=int(payload.get("stored_points") or payload.get("chunk_count") or 0),
        file_size=int(payload.get("file_size") or 0),
        modified_at=string_value(payload.get("modified_at")),
        source_fingerprint=string_value(payload.get("source_fingerprint")),
        status=string_value(payload.get("status")) or "indexed",
        error_message=string_value(payload.get("error_message")) or None,
        artifact_paths=[
            item for item in artifact_paths if isinstance(item, str)
        ]
        if isinstance(artifact_paths, list)
        else [],
    )


def string_value(value: Any) -> str:
    """Return a stripped string value or an empty string."""
    return value.strip() if isinstance(value, str) else ""
