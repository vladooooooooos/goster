"""Indexator adapter for shared Qdrant vector storage."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import models  # noqa: E402

from shared.vector_store import (  # noqa: E402
    GostBlockVector,
    QdrantVectorStore,
    QdrantVectorStoreConfig,
    make_gost_block_point,
)

from app.core.blocks import StructuredBlock
from app.services.embedding_service import BlockEmbedding, BlockEmbeddingRun
from app.utils.config import StorageConfig


@dataclass(frozen=True)
class QdrantStorageRun:
    """Debug metadata for one Qdrant server upsert run."""

    collection_name: str
    endpoint: str
    stored_blocks: int
    embedding_dimension: int
    elapsed_seconds: float


@dataclass(frozen=True)
class QdrantDocumentDeletionResult:
    """Per-document Qdrant server deletion result."""

    document_id: str
    removed_points: int
    success: bool
    error_message: str | None = None


@dataclass(frozen=True)
class QdrantDocumentDeletionRun:
    """Summary of Qdrant server deletion by document id."""

    collection_name: str
    endpoint: str
    requested_documents: int
    removed_documents: int
    skipped_documents: int
    removed_points: int
    elapsed_seconds: float
    results: list[QdrantDocumentDeletionResult]


@dataclass(frozen=True)
class QdrantClearRun:
    """Summary of a Qdrant server collection clear operation."""

    collection_name: str
    endpoint: str
    collection_existed: bool
    removed_points: int
    elapsed_seconds: float


class QdrantStore:
    """Qdrant server store for vectorized structured blocks."""

    def __init__(
        self,
        config: QdrantVectorStoreConfig,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        vector_store_kwargs: dict[str, Any] = {}
        if client_factory is not None:
            vector_store_kwargs["client_factory"] = client_factory
        self.vector_store = QdrantVectorStore(
            config,
            **vector_store_kwargs,
        )
        self.endpoint = self.vector_store.endpoint
        self.collection_name = self.vector_store.collection_name
        self.distance_metric = self.vector_store.distance_metric
        self.client = self.vector_store.client

    @classmethod
    def from_config(cls, config: StorageConfig, app_root: Path) -> "QdrantStore":
        """Create a Qdrant server store from application config."""
        return cls(
            QdrantVectorStoreConfig(
                collection_name=config.collection_name,
                distance_metric=config.distance_metric,
                url=config.url,
                host=config.host,
                port=config.port,
                https=config.https,
                api_key=config.api_key,
                timeout_seconds=config.timeout_seconds,
            )
        )

    def ensure_collection(self, embedding_dimension: int) -> None:
        """Create the collection if it does not already exist."""
        self.vector_store.ensure_collection(embedding_dimension)

    def upsert_block_embeddings(
        self,
        blocks: list[StructuredBlock],
        embedding_run: BlockEmbeddingRun,
        indexed_at: str | None = None,
    ) -> QdrantStorageRun:
        """Store structured blocks and their embeddings in Qdrant server."""
        resolved_indexed_at = indexed_at or utc_now_iso()
        blocks_by_id = {block.id: block for block in blocks}
        points = [
            make_gost_block_point(
                make_gost_block_vector(blocks_by_id[embedding.block_id], embedding, resolved_indexed_at)
            )
            for embedding in embedding_run.embeddings
            if embedding.block_id in blocks_by_id
        ]

        storage_run = self.vector_store.upsert_points(
            points=points,
            embedding_dimension=embedding_run.embedding_dimension,
        )
        return QdrantStorageRun(
            collection_name=storage_run.collection_name,
            endpoint=storage_run.endpoint,
            stored_blocks=storage_run.stored_points,
            embedding_dimension=storage_run.embedding_dimension,
            elapsed_seconds=storage_run.elapsed_seconds,
        )

    def count_document_points(self, document_id: str) -> int:
        """Return point count for one document id without failing on missing collections."""
        if not self.vector_store.call_qdrant(self.client.collection_exists, self.collection_name):
            return 0

        result = self.vector_store.call_qdrant(
            self.client.count,
            collection_name=self.collection_name,
            count_filter=make_document_filter([document_id]),
            exact=True,
        )
        return int(getattr(result, "count", 0) or 0)

    def delete_documents(self, document_ids: list[str]) -> QdrantDocumentDeletionRun:
        """Delete vector points for specific document ids."""
        start_time = perf_counter()
        unique_document_ids = unique_nonempty(document_ids)
        results: list[QdrantDocumentDeletionResult] = []

        if not self.vector_store.call_qdrant(self.client.collection_exists, self.collection_name):
            return QdrantDocumentDeletionRun(
                collection_name=self.collection_name,
                endpoint=self.endpoint,
                requested_documents=len(unique_document_ids),
                removed_documents=0,
                skipped_documents=len(unique_document_ids),
                removed_points=0,
                elapsed_seconds=perf_counter() - start_time,
                results=[
                    QdrantDocumentDeletionResult(document_id=document_id, removed_points=0, success=True)
                    for document_id in unique_document_ids
                ],
            )

        for document_id in unique_document_ids:
            try:
                point_count = self.count_document_points(document_id)
                if point_count:
                    self.vector_store.call_qdrant(
                        self.client.delete,
                        collection_name=self.collection_name,
                        points_selector=models.FilterSelector(filter=make_document_filter([document_id])),
                        wait=True,
                    )
                results.append(
                    QdrantDocumentDeletionResult(
                        document_id=document_id,
                        removed_points=point_count,
                        success=True,
                    )
                )
            except Exception as error:
                results.append(
                    QdrantDocumentDeletionResult(
                        document_id=document_id,
                        removed_points=0,
                        success=False,
                        error_message=str(error),
                    )
                )

        removed_documents = sum(1 for result in results if result.success and result.removed_points > 0)
        skipped_documents = sum(1 for result in results if result.success and result.removed_points == 0)
        return QdrantDocumentDeletionRun(
            collection_name=self.collection_name,
            endpoint=self.endpoint,
            requested_documents=len(unique_document_ids),
            removed_documents=removed_documents,
            skipped_documents=skipped_documents,
            removed_points=sum(result.removed_points for result in results),
            elapsed_seconds=perf_counter() - start_time,
            results=results,
        )

    def clear_all(self) -> QdrantClearRun:
        """Remove the configured collection while preserving the shared data root."""
        start_time = perf_counter()
        collection_existed = self.vector_store.call_qdrant(self.client.collection_exists, self.collection_name)
        removed_points = 0

        if collection_existed:
            count_result = self.vector_store.call_qdrant(
                self.client.count,
                collection_name=self.collection_name,
                exact=True,
            )
            removed_points = int(getattr(count_result, "count", 0) or 0)
            self.vector_store.call_qdrant(self.client.delete_collection, collection_name=self.collection_name)

        return QdrantClearRun(
            collection_name=self.collection_name,
            endpoint=self.endpoint,
            collection_existed=collection_existed,
            removed_points=removed_points,
            elapsed_seconds=perf_counter() - start_time,
        )

    def close(self) -> None:
        """Close the Qdrant client."""
        self.vector_store.close()


def make_gost_block_vector(block: StructuredBlock, embedding: BlockEmbedding, indexed_at: str) -> GostBlockVector:
    """Convert Indexator block and embedding objects to the shared storage model."""
    return GostBlockVector(
        block_id=block.id,
        doc_id=block.doc_id,
        document_id=block.doc_id,
        file_name=block.file_name,
        file_path=block.file_path,
        block_type=block.block_type,
        page_number=block.page_number,
        text=block.text,
        embedding_text=embedding.text,
        vector=embedding.vector,
        section_path=block.section_path,
        reading_order=block.reading_order,
        indexed_at=indexed_at,
        label=block.label,
        context_text=block.context_text,
        bbox=block.bbox,
    )


def make_document_filter(document_ids: list[str]) -> models.Filter:
    """Build a compatibility filter for current and future document id payload keys."""
    ids = unique_nonempty(document_ids)
    return models.Filter(
        should=[
            models.FieldCondition(key="document_id", match=models.MatchAny(any=ids)),
            models.FieldCondition(key="doc_id", match=models.MatchAny(any=ids)),
        ]
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


def utc_now_iso() -> str:
    """Return the current UTC timestamp for payload metadata."""
    return datetime.now(UTC).isoformat()
