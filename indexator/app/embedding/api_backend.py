"""API embedding backend intentionally not implemented for the current MVP scope."""

from __future__ import annotations


class BgeM3ApiEmbeddingBackend:
    """Disabled placeholder to keep old imports explicit during the local-only phase."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("API embeddings are outside the current local BAAI/bge-m3 MVP scope.")
