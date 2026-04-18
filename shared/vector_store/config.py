"""Shared Qdrant vector store configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QdrantVectorStoreConfig:
    """Connection and collection settings for a Qdrant server."""

    collection_name: str
    distance_metric: str = "Cosine"
    url: str = "http://127.0.0.1:6333"
    host: str = "127.0.0.1"
    port: int = 6333
    https: bool = False
    api_key: str | None = None
    timeout_seconds: float = 5.0

    @property
    def endpoint(self) -> str:
        """Return the effective Qdrant endpoint for logs and errors."""
        if self.url.strip():
            return self.url.strip()
        scheme = "https" if self.https else "http"
        return f"{scheme}://{self.host}:{self.port}"
