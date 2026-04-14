"""Shared vector store data models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VectorPoint:
    """Generic vector point ready to be written into a vector store."""

    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class VectorSearchResult:
    """Generic vector search result returned from a vector store."""

    id: Any
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class VectorStorageRun:
    """Debug metadata for one vector store write run."""

    collection_name: str
    local_path: Path
    stored_points: int
    embedding_dimension: int
    elapsed_seconds: float


@dataclass(frozen=True)
class GostBlockVector:
    """GOST document block plus embedding data prepared for vector storage."""

    block_id: str
    doc_id: str
    document_id: str
    file_name: str
    file_path: Path
    block_type: str
    page_number: int
    text: str
    embedding_text: str
    vector: list[float]
    section_path: list[str]
    reading_order: int
    indexed_at: str
    label: str | None = None
    context_text: str | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class GostPayloadFields:
    """Common GOST payload fields used by retrieval consumers."""

    block_id: str
    text: str
    retrieval_text: str
    source_file: str
    page_start: int | None
    page_end: int | None
    section_path: list[str]
    document_id: str | None
    block_type: str | None
    label: str | None
