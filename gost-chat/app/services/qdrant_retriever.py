from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.vector_store import (  # noqa: E402
    QdrantVectorStoreConfig,
    QdrantVectorStore,
    VectorSearchResult,
    parse_gost_payload,
)

from app.services.local_embedding_service import LocalEmbeddingService
from app.services.retrieval_types import RetrievedBlock

logger = logging.getLogger(__name__)


class QdrantRetriever:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_host: str,
        qdrant_port: int,
        qdrant_https: bool,
        qdrant_api_key: str | None,
        qdrant_timeout_seconds: float,
        collection_name: str,
        embedding_service: LocalEmbeddingService,
        vector_store_factory: Callable[[QdrantVectorStoreConfig], QdrantVectorStore] | None = None,
    ) -> None:
        self.config = QdrantVectorStoreConfig(
            collection_name=collection_name,
            url=qdrant_url,
            host=qdrant_host,
            port=qdrant_port,
            https=qdrant_https,
            api_key=qdrant_api_key,
            timeout_seconds=qdrant_timeout_seconds,
        )
        self.collection_name = self.config.collection_name
        self.endpoint = self.config.endpoint
        self.embedding_service = embedding_service
        self._vector_store_factory = vector_store_factory or QdrantVectorStore
        self._vector_store: QdrantVectorStore | None = None

    @property
    def vector_store(self) -> QdrantVectorStore:
        if self._vector_store is None:
            logger.info("Opening Qdrant server collection %s at %s.", self.collection_name, self.endpoint)
            self._vector_store = self._vector_store_factory(self.config)
        return self._vector_store

    @property
    def client(self) -> Any:
        return self.vector_store.client

    def search(self, query: str, top_k: int) -> tuple[list[RetrievedBlock], dict[str, Any]]:
        vector = self.embedding_service.embed_query(query)
        candidates = self.search_by_vector(vector=vector, top_k=top_k)
        logger.info("Qdrant retrieved %s candidate block(s).", len(candidates))
        return candidates, {
            "backend": "qdrant",
            "top_k": top_k,
            "collection_name": self.collection_name,
            "endpoint": self.endpoint,
            "embedding_model": self.embedding_service.settings.model_name,
            "embedding_device": self.embedding_service.device,
        }

    def search_by_vector(self, vector: list[float], top_k: int) -> list[RetrievedBlock]:
        """Search Qdrant with an already embedded query vector."""
        points = self.vector_store.search(vector=vector, top_k=top_k)
        return [block for point in points if (block := retrieved_block_from_point(point))]

    def close(self) -> None:
        if self._vector_store is not None:
            self._vector_store.close()


def retrieved_block_from_point(point: VectorSearchResult) -> RetrievedBlock | None:
    if not point.payload:
        return None

    fields = parse_gost_payload(point.payload, fallback_id=point.id)
    return RetrievedBlock(
        block_id=fields.block_id,
        text=fields.text,
        retrieval_text=fields.retrieval_text,
        source_file=fields.source_file,
        page=fields.page_start,
        section_path=fields.section_path,
        retrieval_score=round(point.score, 6),
        payload=dict(point.payload),
        document_id=fields.document_id,
        page_start=fields.page_start,
        page_end=fields.page_end,
        block_type=fields.block_type,
        label=fields.label,
    )
