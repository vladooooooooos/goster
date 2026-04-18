from __future__ import annotations

import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.services.qdrant_retriever import QdrantRetriever
from shared.vector_store import VectorSearchResult


class GostChatQdrantServerConfigTest(unittest.TestCase):
    def test_settings_expose_qdrant_server_defaults(self) -> None:
        settings = Settings(_env_file=None)

        self.assertEqual(settings.qdrant_url, "http://127.0.0.1:6333")
        self.assertEqual(settings.qdrant_host, "127.0.0.1")
        self.assertEqual(settings.qdrant_port, 6333)
        self.assertFalse(settings.qdrant_https)
        self.assertEqual(settings.qdrant_timeout_seconds, 5.0)
        self.assertIsNone(settings.qdrant_api_key)
        self.assertFalse(hasattr(settings, "qdrant_local_path"))

    def test_retriever_metadata_reports_endpoint(self) -> None:
        retriever = QdrantRetriever(
            qdrant_url="http://127.0.0.1:6333",
            qdrant_host="127.0.0.1",
            qdrant_port=6333,
            qdrant_https=False,
            qdrant_api_key=None,
            qdrant_timeout_seconds=5.0,
            collection_name="gost_blocks",
            embedding_service=FakeEmbeddingService(),
            vector_store_factory=lambda config: FakeVectorStore(config.endpoint),
        )

        blocks, info = retriever.search("query", top_k=3)

        self.assertEqual(blocks[0].block_id, "block-1")
        self.assertEqual(info["backend"], "qdrant")
        self.assertEqual(info["endpoint"], "http://127.0.0.1:6333")
        self.assertEqual(info["collection_name"], "gost_blocks")
        self.assertNotIn("local_path", info)


class FakeEmbeddingSettings:
    model_name = "fake-embedding"


class FakeEmbeddingService:
    settings = FakeEmbeddingSettings()
    device = "cpu"

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def search(self, vector: list[float], top_k: int) -> list[VectorSearchResult]:
        return [
            VectorSearchResult(
                id="block-1",
                score=0.9,
                payload={
                    "block_id": "block-1",
                    "text": "Retrieved text",
                    "source_file": "source.pdf",
                    "page_start": 1,
                    "section_path": [],
                    "document_id": "doc-1",
                },
            )
        ]

    def close(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
