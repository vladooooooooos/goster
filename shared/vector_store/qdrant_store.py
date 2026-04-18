"""Shared Qdrant server vector store infrastructure."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from qdrant_client import QdrantClient, models

from .config import QdrantVectorStoreConfig
from .models import VectorPoint, VectorSearchResult, VectorStorageRun


class QdrantVectorStore:
    """Reusable Qdrant server vector store for GOSTer apps."""

    def __init__(
        self,
        config: QdrantVectorStoreConfig,
        client_factory: Callable[..., Any] = QdrantClient,
    ) -> None:
        self.config = config
        self.collection_name = config.collection_name
        self.distance_metric = config.distance_metric
        self.endpoint = config.endpoint
        self.client = client_factory(**make_client_kwargs(config))

    def ensure_collection(self, embedding_dimension: int) -> None:
        """Create the collection if it does not already exist."""
        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be greater than zero.")
        if self.call_qdrant(self.client.collection_exists, self.collection_name):
            return

        self.call_qdrant(
            self.client.create_collection,
            collection_name=self.collection_name,
            vectors_config=make_vectors_config(
                embedding_dimension,
                self.distance_metric,
                vectors_on_disk=self.config.vectors_on_disk,
            ),
            **make_collection_optimization_kwargs(self.config),
        )

    def upsert_points(self, points: list[VectorPoint], embedding_dimension: int) -> VectorStorageRun:
        """Write vector points into the configured Qdrant collection."""
        if not points:
            return VectorStorageRun(
                collection_name=self.collection_name,
                endpoint=self.endpoint,
                stored_points=0,
                embedding_dimension=embedding_dimension,
                elapsed_seconds=0.0,
            )

        self.ensure_collection(embedding_dimension)
        start_time = perf_counter()
        for batch in iter_batches(points, self.config.upsert_batch_size):
            self.call_qdrant(
                self.client.upsert,
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                    for point in batch
                ],
                wait=True,
            )
        elapsed_seconds = perf_counter() - start_time

        return VectorStorageRun(
            collection_name=self.collection_name,
            endpoint=self.endpoint,
            stored_points=len(points),
            embedding_dimension=embedding_dimension,
            elapsed_seconds=elapsed_seconds,
        )

    def search(self, vector: list[float], top_k: int, with_payload: bool = True) -> list[VectorSearchResult]:
        """Search the configured Qdrant collection by vector."""
        if not self.call_qdrant(self.client.collection_exists, self.collection_name):
            return []

        if hasattr(self.client, "query_points"):
            response = self.call_qdrant(
                self.client.query_points,
                collection_name=self.collection_name,
                query=vector,
                limit=top_k,
                with_payload=with_payload,
            )
            points = list(getattr(response, "points", response))
        else:
            points = list(
                self.call_qdrant(
                    self.client.search,
                    collection_name=self.collection_name,
                    query_vector=vector,
                    limit=top_k,
                    with_payload=with_payload,
                )
            )

        return [make_search_result(point) for point in points]

    def close(self) -> None:
        """Close the Qdrant client."""
        self.client.close()

    def call_qdrant(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a Qdrant operation and wrap connection failures with endpoint context."""
        try:
            return operation(*args, **kwargs)
        except QdrantServerConnectionError:
            raise
        except Exception as error:
            raise QdrantServerConnectionError(self.endpoint, error) from error


class QdrantServerConnectionError(RuntimeError):
    """Raised when the configured Qdrant server cannot be reached."""

    def __init__(self, endpoint: str, cause: Exception) -> None:
        super().__init__(f"Qdrant server is not reachable at {endpoint}: {cause}")
        self.endpoint = endpoint
        self.cause = cause


def make_client_kwargs(config: QdrantVectorStoreConfig) -> dict[str, Any]:
    """Build qdrant-client keyword arguments for server mode."""
    kwargs: dict[str, Any]
    if config.url.strip():
        kwargs = {"url": config.url.strip()}
    else:
        kwargs = {"host": config.host, "port": config.port, "https": config.https}

    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.timeout_seconds > 0:
        kwargs["timeout"] = config.timeout_seconds
    return kwargs


def iter_batches(points: list[VectorPoint], batch_size: int) -> list[list[VectorPoint]]:
    """Split points into non-empty batches for Qdrant request size control."""
    resolved_batch_size = max(1, batch_size)
    return [
        points[start : start + resolved_batch_size]
        for start in range(0, len(points), resolved_batch_size)
    ]


def make_vectors_config(
    embedding_dimension: int,
    distance_metric: str,
    vectors_on_disk: bool = False,
) -> models.VectorParams:
    """Build vector params in one place for future server-side storage tuning."""
    return models.VectorParams(
        size=embedding_dimension,
        distance=resolve_distance(distance_metric),
        on_disk=vectors_on_disk,
    )


def make_collection_optimization_kwargs(config: QdrantVectorStoreConfig) -> dict[str, Any]:
    """Build optional server-side collection storage optimization settings."""
    kwargs: dict[str, Any] = {}
    if config.quantization_enabled and config.quantization_mode.strip().lower() == "scalar":
        kwargs["quantization_config"] = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=config.quantized_vectors_always_ram,
            )
        )
    return kwargs


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
