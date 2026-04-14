"""Shared local Qdrant vector store infrastructure."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from qdrant_client import QdrantClient, models

from .config import QdrantVectorStoreConfig
from .models import VectorPoint, VectorSearchResult, VectorStorageRun


class QdrantVectorStore:
    """Reusable local Qdrant vector store for GOSTer apps."""

    def __init__(self, config: QdrantVectorStoreConfig) -> None:
        self.local_path = config.local_path
        self.collection_name = config.collection_name
        self.distance_metric = config.distance_metric
        self.local_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.local_path))

    def ensure_collection(self, embedding_dimension: int) -> None:
        """Create the collection if it does not already exist."""
        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be greater than zero.")
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dimension,
                distance=resolve_distance(self.distance_metric),
            ),
        )

    def upsert_points(self, points: list[VectorPoint], embedding_dimension: int) -> VectorStorageRun:
        """Write vector points into the configured Qdrant collection."""
        if not points:
            return VectorStorageRun(
                collection_name=self.collection_name,
                local_path=self.local_path,
                stored_points=0,
                embedding_dimension=embedding_dimension,
                elapsed_seconds=0.0,
            )

        self.ensure_collection(embedding_dimension)
        start_time = perf_counter()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in points
            ],
            wait=True,
        )
        elapsed_seconds = perf_counter() - start_time

        return VectorStorageRun(
            collection_name=self.collection_name,
            local_path=self.local_path,
            stored_points=len(points),
            embedding_dimension=embedding_dimension,
            elapsed_seconds=elapsed_seconds,
        )

    def search(self, vector: list[float], top_k: int, with_payload: bool = True) -> list[VectorSearchResult]:
        """Search the configured Qdrant collection by vector."""
        if not self.client.collection_exists(self.collection_name):
            return []

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=top_k,
                with_payload=with_payload,
            )
            points = list(getattr(response, "points", response))
        else:
            points = list(
                self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    limit=top_k,
                    with_payload=with_payload,
                )
            )

        return [make_search_result(point) for point in points]

    def close(self) -> None:
        """Close the local Qdrant client and release the storage lock."""
        self.client.close()


def make_search_result(point: Any) -> VectorSearchResult:
    """Convert a qdrant-client point object into a shared search result."""
    payload = getattr(point, "payload", None)
    return VectorSearchResult(
        id=getattr(point, "id", None),
        score=float(getattr(point, "score", 0.0) or 0.0),
        payload=dict(payload) if isinstance(payload, dict) else {},
    )


def resolve_distance(distance_metric: str) -> models.Distance:
    """Resolve a config distance metric to a Qdrant distance enum."""
    normalized = distance_metric.strip().lower()
    if normalized == "dot":
        return models.Distance.DOT
    if normalized == "euclid":
        return models.Distance.EUCLID
    return models.Distance.COSINE
